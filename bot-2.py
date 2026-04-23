import os
import asyncio
import re
import json
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

class SimpleRAGBot:
    def __init__(self, gigachat_token: str, telegram_token: str):
        self.gigachat_token = gigachat_token
        
        # Модель для эмбеддингов (поддерживает русский и английский)
        self.embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Данные документов
        self.documents = []
        self.chunks = []
        self.embeddings = None

        # Логирование для RAGAS
        self.interaction_log = []
        self.log_file = "ragas_log.json"
        self.load_existing_log()
        
        # Загрузка документов
        self.load_documents_from_folder("/Users/kseniatebenkova/Desktop/new data/md")
        
        # Telegram бот
        self.bot = Bot(token=telegram_token)

        self.dp = Dispatcher()
        self.setup_handlers()

    # Методы для логирования
    def load_existing_log(self):
        """Загружает существующий лог если есть"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.interaction_log = json.load(f)
                print(f"Loaded {len(self.interaction_log)} existing log entries")
        except Exception as e:
            print(f"Could not load existing log: {e}")
            self.interaction_log = []
    
    def log_interaction(self, question: str, answer: str, contexts: list):
        """Логирует одно взаимодействие для RAGAS"""
        # Конвертируем numpy.float32 в обычный float
        converted_scores = []
        for c in contexts:
            if isinstance(c['score'], (np.floating, np.float32, np.float64)):
                converted_scores.append(float(c['score']))
            else:
                converted_scores.append(c['score'])
    
        log_entry = {
            'question': question,
            'answer': answer,
            'contexts': [c['text'] for c in contexts]
        }
    
        self.interaction_log.append(log_entry)
    
        # Сохраняем в файл
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.interaction_log, f, ensure_ascii=False, indent=2)
            print(f"Logged interaction #{len(self.interaction_log)}")
        except Exception as e:
            print(f"Failed to save log: {e}")
    
    def export_for_ragas(self, output_file: str = "ragas_dataset.json"):
        """Экспортирует логи в формат для RAGAS"""
        ragas_format = {
            'question': [],
            'answer': [],
            'contexts': []
        }
        
        for log in self.interaction_log:
            ragas_format['question'].append(log['question'])
            ragas_format['answer'].append(log['answer'])
            ragas_format['contexts'].append(log['contexts'])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ragas_format, f, ensure_ascii=False, indent=2)
        
        print(f"Exported {len(self.interaction_log)} interactions to {output_file}")
        return output_file
    
    def detect_language(self, text: str) -> str:
        """Определяет язык текста (русский или английский)"""
        ru_chars = sum(1 for c in text if 'а' <= c.lower() <= 'я')
        en_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        
        if ru_chars > 0:
            return 'ru'
        elif en_chars > 0:
            return 'en'
        else:
            return 'ru'  # по умолчанию
    
    def get_system_prompt(self, lang: str) -> str:
        """Возвращает системный промпт на нужном языке"""
        prompts = {
            'ru': """Ты - помощник, который отвечает на вопросы на основе технической документации.
Отвечай ТОЛЬКО используя информацию из предоставленных документов.
Если ответа нет в документах - скажи "Не могу найти ответ в документах".
Отвечай на русском языке. Для технических терминов используй оригинальные названия.""",
            
            'en': """You are an assistant that answers questions based on technical documentation.
Answer ONLY using information from the provided documents.
If the answer is not in the documents - say "I cannot find the answer in the documents".
Answer in English."""
        }
        return prompts.get(lang, prompts['ru'])
    
    def clean_markdown(self, text: str) -> str:
        """Очищает Markdown разметку, оставляя читаемый текст"""
        # Удаляем HTML теги
        text = re.sub(r'<[^>]+>', '', text)
        
        # Удаляем блоки кода
        text = re.sub(r'```[\s\S]*?```', '[CODE BLOCK]', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Удаляем заголовки (#)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Удаляем жирный текст и курсив
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Удаляем ссылки [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Удаляем изображения ![alt](url)
        text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '[IMAGE]', text)
        
        # Удаляем горизонтальные линии
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        
        # Удаляем маркированные списки
        text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
        
        # Удаляем нумерованные списки
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Удаляем цитаты
        text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def extract_metadata_from_markdown(self, content: str, filename: str) -> dict:
        """Извлекает метаданные из Markdown файла"""
        metadata = {
            "title": filename.replace('.md', '').replace('_', ' ').title(),
            "author": "Technical Documentation",
            "sections": []
        }
        
        # Ищем заголовки для структуры документа
        headers = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
        if headers:
            metadata["sections"] = headers[:5]  # Первые 5 заголовков
        
        # Ищем YAML frontmatter если есть
        frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            
            # Извлекаем title
            title_match = re.search(r'title:\s*["\']?(.+?)["\']?(?:\n|$)', frontmatter)
            if title_match:
                metadata["title"] = title_match.group(1)
            
            # Извлекаем author
            author_match = re.search(r'author:\s*["\']?(.+?)["\']?(?:\n|$)', frontmatter)
            if author_match:
                metadata["author"] = author_match.group(1)
        
        return metadata
    
    def load_documents_from_folder(self, folder_path: str):
        """Загружает все Markdown файлы из папки"""
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не найдена")
            return
        
        for filename in os.listdir(folder_path):
            if filename.endswith(('.md', '.markdown')):
                self.load_markdown(os.path.join(folder_path, filename))
        
        if not self.documents:
            print("Не найдено Markdown файлов")
        else:
            print(f"Загружено {len(self.documents)} Markdown файлов")
        
        self.create_chunks_and_embeddings()
    
    def load_markdown(self, file_path: str):
        """Загружает один Markdown файл"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Извлекаем метаданные
            metadata = self.extract_metadata_from_markdown(content, os.path.basename(file_path))
            
            # Очищаем Markdown разметку
            cleaned_text = self.clean_markdown(content)
            
            self.documents.append({
                "text": cleaned_text,
                "raw_content": content,  # Сохраняем оригинал для отладки
                "title": metadata["title"],
                "author": metadata["author"],
                "sections": metadata["sections"],
                "file_path": file_path
            })
            
        except Exception as e:
            print(f"Ошибка при загрузке файла {file_path}: {e}")
    
    def create_chunks_and_embeddings(self, chunk_size: int = 1000):
        """Разбивает документы на чанки и создает эмбеддинги"""
        for doc in self.documents:
            text = doc["text"]
            
            # Разделяем по двойным переносам строк (параграфы)
            paragraphs = text.split('\n\n')
            
            for para in paragraphs:
                if para.strip():
                    if len(para) > chunk_size:
                        # Разбиваем длинные параграфы на предложения
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        current_chunk = ""
                        
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) < chunk_size:
                                current_chunk += sentence + " "
                            else:
                                if current_chunk:
                                    self.chunks.append({
                                        "text": current_chunk.strip(),
                                        "title": doc["title"],
                                        "author": doc["author"]
                                    })
                                current_chunk = sentence + " "
                        
                        if current_chunk:
                            self.chunks.append({
                                "text": current_chunk.strip(),
                                "title": doc["title"],
                                "author": doc["author"]
                            })
                    else:
                        self.chunks.append({
                            "text": para.strip(),
                            "title": doc["title"],
                            "author": doc["author"]
                        })
        
        # Создаем эмбеддинги
        if self.chunks:
            chunk_texts = [chunk["text"] for chunk in self.chunks]
            self.embeddings = self.embed_model.encode(chunk_texts, convert_to_numpy=True)
            print(f"Создано {len(self.chunks)} чанков")
    
    def find_relevant_chunks(self, query: str, top_k: int = 20):
        """Находит наиболее релевантные чанки"""
        if not self.chunks:
            return []
        
        # Эмбеддинг запроса
        query_embedding = self.embed_model.encode([query])
        
        # Поиск похожих чанков
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.2:
                chunk = self.chunks[idx]
                results.append({
                    "text": chunk["text"],
                    "title": chunk["title"],
                    "author": chunk["author"],
                    "score": similarities[idx]
                })
        
        return results
    
    async def ask_gigachat(self, question: str, context_chunks: list, lang: str):
        """Задает вопрос GigaChat с контекстом"""
        context = "\n\n".join([
            f"[Document: {chunk['title']}]\n{chunk['text']}"
            for chunk in context_chunks
        ])
    
        system_prompt = self.get_system_prompt(lang)
    
        if lang == 'ru':
            user_prompt = f"""Ты - ассистент по технической документации. Твоя задача - отвечать на вопросы, используя ТОЛЬКО информацию из предоставленного контекста.

    ПРАВИЛА РАБОТЫ:
    1. Внимательно прочитай КОНТЕКСТ ниже
    2. Если в КОНТЕКСТЕ есть информация, отвечающая на вопрос - предоставь полный и точный ответ, основанный ТОЛЬКО на контексте
    3. Цитируй технические детали дословно из контекста
    4. Если в КОНТЕКСТЕ действительно нет информации для ответа - скажи "Информация по данному вопросу отсутствует в документации"
    5. Не добавляй информацию из своих знаний, если её нет в контексте

    Вопрос: {question}

    КОНТЕКСТ ИЗ ДОКУМЕНТАЦИИ:
    {context}

    Если информация есть - дай подробный ответ. Если информации нет - честно скажи об этом."""
        else:
            user_prompt = f"""You are a technical documentation assistant. Your task is to answer questions using ONLY information from the provided context.

    WORKING RULES:
    1. Carefully read the CONTEXT below
    2. If the CONTEXT contains information answering the question - provide a complete and accurate answer based ONLY on the context
    3. Quote technical details verbatim from the context
    4. If the CONTEXT truly lacks information to answer - say "Information on this question is not found in the documentation"
    5. Do not add information from your knowledge if it's not in the context

    Question: {question}

    DOCUMENTATION CONTEXT:
    {context}

    If information exists - provide a detailed answer. If not - honestly say so."""
    
        try:
            # Создаем клиент GigaChat
            giga = GigaChat(
                credentials=self.gigachat_token,
                scope="GIGACHAT_API_PERS",
                model="GigaChat-Pro",
                verify_ssl_certs=False
            )
        
            # Выполняем запрос в отдельном потоке чтобы не блокировать event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: giga.chat(Chat(
                    messages=[
                        Messages(role=MessagesRole.SYSTEM, content=system_prompt),
                        Messages(role=MessagesRole.USER, content=user_prompt)
                    ],
                    temperature=0.2,
                    max_tokens=3000
                ))
            )
        
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"GigaChat error: {e}")
            if lang == 'ru':
                return f"Ошибка при генерации ответа: {str(e)}"
            else:
                return f"Error generating answer: {str(e)}"

    def setup_handlers(self):
        """Настраивает обработчики команд"""
        @self.dp.message(Command("start"))
        async def start(message: Message):
            text = (
                f"🤖 RAG-bot для технической документации готов!\n"
                f"📚 Загружено документов: {len(self.documents)}\n"
                f"📄 Количество фрагментов: {len(self.chunks)}\n"
                f"📊 Взаимодействий в логе: {len(self.interaction_log)}\n\n"
                f"Задайте вопрос по технической документации.\n\n"
            )
            await message.answer(text)
        
        @self.dp.message(Command("list"))
        async def list_docs(message: Message):
            if not self.documents:
                await message.answer("Документы не загружены")
                return
            
            docs_list = []
            for doc in self.documents:
                sections_info = ""
                if doc.get('sections'):
                    sections_info = f"\n   Разделы: {', '.join(doc['sections'][:3])}"
                docs_list.append(f"• {doc['title']}{sections_info}")
            
            docs_text = "\n".join(docs_list)
            await message.answer(f"📚 Загруженные документы:\n{docs_text}")
        
        @self.dp.message()
        async def handle_question(message: Message):
            question = message.text.strip()
            if not question:
                return
            
            # Определяем язык вопроса
            lang = self.detect_language(question)
            
            # Показываем статус на нужном языке
            if lang == 'ru':
                status = await message.answer("🔍 Ищу информацию в технической документации...")
            else:
                status = await message.answer("🔍 Searching technical documentation...")
            
            # Поиск релевантных чанков
            relevant_chunks = self.find_relevant_chunks(question)
            
            print(f"\n=== DEBUG ===")
            print(f"Question: {question}")
            print(f"Found chunks: {len(relevant_chunks)}")
            for i, chunk in enumerate(relevant_chunks[:3]):
                print(f"Chunk {i+1} (score: {chunk['score']:.3f}):")
                print(f"Text preview: {chunk['text'][:200]}...")
            print(f"=============\n")
            
            if not relevant_chunks:
                if lang == 'ru':
                    await status.edit_text("❌ Не найдено подходящей информации в документах")
                else:
                    await status.edit_text("❌ No relevant information found in documents")
                return
            
            # Генерация ответа
            if lang == 'ru':
                await status.edit_text("✍️ Генерирую ответ на основе документации...")
            else:
                await status.edit_text("✍️ Generating answer based on documentation...")
            
            answer = await self.ask_gigachat(question, relevant_chunks, lang)

            # Логирование взаимодействия
            self.log_interaction(question, answer, relevant_chunks)
            
            print(f"\n=== GIGACHAT RESPONSE ===")
            print(f"Answer length: {len(answer)}")
            print(f"Full answer:\n{answer}")
            print(f"==========================\n")
            
            # Формируем финальный ответ без источников
            final_answer = answer
            
            # Разбиваем длинные сообщения
            if len(final_answer) > 4096:
                parts = [final_answer[i:i+4096] for i in range(0, len(final_answer), 4096)]
                await status.delete()
                for part in parts:
                    await message.answer(part, parse_mode="Markdown")
            else:
                await status.edit_text(final_answer, parse_mode="Markdown")
    
    async def run(self):
        """Запускает бота"""
        print("Бот для работы с технической документацией запущен!")
        await self.dp.start_polling(self.bot)

# Запуск бота
async def main():
    GIGACHAT_TOKEN = ""
    TELEGRAM_TOKEN = ""
    
    bot = SimpleRAGBot(GIGACHAT_TOKEN, TELEGRAM_TOKEN)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
