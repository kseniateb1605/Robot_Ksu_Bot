import os
import pypdf
import asyncio
import numpy as np
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
        
        # Загрузка документов
        self.load_documents_from_folder("data")
        
        # Telegram бот
        self.bot = Bot(token=telegram_token)
        self.dp = Dispatcher()
        self.setup_handlers()
    
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
            'ru': """Ты - помощник, который отвечает на вопросы на основе документов.
Отвечай ТОЛЬКО используя информацию из предоставленных документов.
Если ответа нет в документах - скажи "Не могу найти ответ в документах".
Отвечай на русском языке.""",
            
            'en': """You are an assistant that answers questions based on documents.
Answer ONLY using information from the provided documents.
If the answer is not in the documents - say "I cannot find the answer in the documents".
Answer in English."""
        }
        return prompts.get(lang, prompts['ru'])
    
    def load_documents_from_folder(self, folder_path: str):
        """Загружает все PDF из папки"""
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не найдена")
            return
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.pdf'):
                self.load_pdf(os.path.join(folder_path, filename))
        
        self.create_chunks_and_embeddings()
        print(f"Загружено {len(self.documents)} документов, {len(self.chunks)} чанков")
    
    def load_pdf(self, file_path: str):
        """Загружает один PDF файл"""
        with open(file_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            
            metadata = pdf.metadata or {}
            title = metadata.get('/Title', os.path.basename(file_path))
            author = metadata.get('/Author', 'Неизвестно')
            
            self.documents.append({
                "text": text,
                "title": title,
                "author": author
            })
    
    def create_chunks_and_embeddings(self, chunk_size: int = 500):
        """Разбивает документы на чанки и создает эмбеддинги"""
        for doc in self.documents:
            text = doc["text"]
            # Разделяем по двойным переносам строк (абзацы)
            paragraphs = text.split('\n\n')
            
            for para in paragraphs:
                if para.strip():
                    if len(para) > chunk_size:
                        sentences = para.split('. ')
                        current_chunk = ""
                        
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) < chunk_size:
                                current_chunk += sentence + ". "
                            else:
                                if current_chunk:
                                    self.chunks.append({
                                        "text": current_chunk.strip(),
                                        "title": doc["title"],
                                        "author": doc["author"]
                                    })
                                current_chunk = sentence + ". "
                        
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
            self.embeddings = self.embed_model.encode(chunk_texts)
    
    def find_relevant_chunks(self, query: str, top_k: int = 3):
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
    
    def ask_gigachat(self, question: str, context_chunks: list, lang: str):
        """Задает вопрос GigaChat с контекстом"""
        context = "\n\n".join([
            f"[Из: {chunk['title']}, автор: {chunk['author']}]\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        system_prompt = self.get_system_prompt(lang)
        
        if lang == 'ru':
            user_prompt = f"""Вопрос: {question}

Контекст из документов:
{context}

Ты - эксперт по химии и робототехнике. Отвечай на вопросы ТОЛЬКО на основе предоставленных документов.

ПРАВИЛА:
1. Если ответ есть в документах - дай полный ответ со всеми деталями
2. Если информации нет в документах - скажи: "Я не могу найти ответ в предоставленных документах"
3. Всегда цитируй источники в конце ответа в формате: "Источник: Название документа, Автор"
4. Не упоминай, что ты используешь документы в тексте ответа
5. Будь точным и информативным, объясняй сложные понятия простым языком
Ответь на русском языке, используя только информацию из контекста."""
        else:
            user_prompt = f"""Question: {question}

Document context:
{context}

You are an expert in chemistry and robotics. Answer questions ONLY based on the provided documents.

RULES:
1. If the answer is in the documents - provide a complete answer with all details
2. If the information is not in the documents - say: "I cannot find the answer in the provided documents"
3. Always cite sources at the end of the answer in the format: "Source: Document Title, Author"
4. Do not mention that you are using documents in the answer text
5. Be accurate and informative, explain complex concepts in simple language
Answer in English, using only information from the context."""
        
        try:
            giga = GigaChat(
                credentials=self.gigachat_token,
                scope="GIGACHAT_API_PERS",
                model="GigaChat-2"
            )
            
            response = giga.chat(Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=system_prompt),
                    Messages(role=MessagesRole.USER, content=user_prompt)
                ],
                temperature=0.1,
                max_tokens=1500
            ))
            
            return response.choices[0].message.content
            
        except Exception as e:
            if lang == 'ru':
                return f"Ошибка при генерации ответа: {str(e)}"
            else:
                return f"Error generating answer: {str(e)}"
    
    def setup_handlers(self):
        """Настраивает обработчики команд"""
        @self.dp.message(Command("start"))
        async def start(message: Message):
            lang = self.detect_language(message.text or "")
            if lang == 'ru':
                text = f"RAG-бот готов к работе!\nЗагружено документов: {len(self.documents)}\nЗадайте вопрос на русском или английском."
            else:
                text = f"RAG-bot is ready!\nLoaded documents: {len(self.documents)}\nAsk a question in Russian or English."
            await message.answer(text)
        
        @self.dp.message(Command("list"))
        async def list_docs(message: Message):
            lang = self.detect_language(message.text or "")
            
            if not self.documents:
                if lang == 'ru':
                    await message.answer("Нет загруженных документов")
                else:
                    await message.answer("No documents loaded")
                return
            
            if lang == 'ru':
                docs_list = "\n".join([f"• {doc['title']} ({doc['author']})" 
                                     for doc in self.documents])
                await message.answer(f"Документы:\n{docs_list}")
            else:
                docs_list = "\n".join([f"• {doc['title']} ({doc['author']})" 
                                     for doc in self.documents])
                await message.answer(f"Documents:\n{docs_list}")
        
        @self.dp.message()
        async def handle_question(message: Message):
            question = message.text.strip()
            
            if not question:
                return
            
            # Определяем язык вопроса
            lang = self.detect_language(question)
            
            # Показываем статус на нужном языке
            if lang == 'ru':
                status = await message.answer("🔍 Ищу информацию в документах...")
            else:
                status = await message.answer("🔍 Searching documents...")
            
            # 1. Поиск релевантных чанков
            relevant_chunks = self.find_relevant_chunks(question)
            
            if not relevant_chunks:
                if lang == 'ru':
                    await status.edit_text("Не найдено подходящей информации в документах")
                else:
                    await status.edit_text("No relevant information found in documents")
                return
            
            # 2. Генерация ответа
            if lang == 'ru':
                await status.edit_text("Генерирую ответ...")
            else:
                await status.edit_text("Generating answer...")
            
            answer = self.ask_gigachat(question, relevant_chunks, lang)
            
            # 3. Добавляем источники
            sources = set()
            for chunk in relevant_chunks:
                sources.add(f"• {chunk['title']} ({chunk['author']})")
            
            sources_text = "\n".join(sources)
            
            if lang == 'ru':
                final_answer = f"{answer}\n\nИсточники:\n{sources_text}"
            else:
                final_answer = f"{answer}\n\nSources:\n{sources_text}"
            
            await status.edit_text(final_answer)
    
    async def run(self):
        """Запускает бота"""
        print("Бот запущен!")
        await self.dp.start_polling(self.bot)

# Запуск бота
async def main():
    GIGACHAT_TOKEN = "_token_gigachat"
    TELEGRAM_TOKEN = "_token_telegram"
    
    bot = SimpleRAGBot(GIGACHAT_TOKEN, TELEGRAM_TOKEN)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
