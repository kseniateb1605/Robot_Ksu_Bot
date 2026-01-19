import os
import pypdf
import asyncio
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    EMBEDDING_MODEL_AVAILABLE = False
    print("sentence-transformers не установлен. Используется упрощенный эмбеддинг.")

class MultiLanguageGigaChatBot:
    def __init__(self, gigachat_token: str, telegram_token: str, chunk_size: int = 500):
        self.gigachat_token = gigachat_token
        self.telegram_token = telegram_token
        self.chunk_size = chunk_size
        
        # Структуры для RAG
        self.loaded_documents = {} 
        self.document_chunks = {} 
        self.chunk_embeddings = {} 
        self.all_chunks = []       
        self.all_embeddings = None  
        self.chunk_to_doc = []     
        
        # Загрузка модели эмбеддингов
        if EMBEDDING_MODEL_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Модель для эмбеддингов загружена")
            except Exception as e:
                print(f"Ошибка загрузки модели эмбеддингов: {e}")
                EMBEDDING_MODEL_AVAILABLE = False
        
        self.preload_and_process_documents()
        
        self.bot = Bot(
            token=telegram_token, 
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        self.dp = Dispatcher()
        self.register_handlers()

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except:
            return 'ru'
    
    def get_language_instruction(self, lang_code: str) -> str:
        instructions = {
            'ru': "Отвечайте полностью, а цитаты указывайте только в конце в формате: \"Источник: Название документа, Автор\".",
            'en': "Answer fully and cite sources only at the end in the format: \"Source: Document Title, Author\"."
        }
        return instructions.get(lang_code, instructions['ru'])
    
    def get_system_prompt(self, lang_code: str) -> str:
        prompts = {
            'ru': """Вы специалист по химии. Ответьте полностью на вопрос, строго на основе предоставленных документов. 
Цитируйте источники только одним списком в конце ответа в формате:
"Источник: Название документа, Автор".
Не вставляйте ссылки после каждого предложения.
Если информации нет в документах - скажите: "Я не могу найти ответ в предоставленных документах".
Будьте точны и информатив.""",
            'en': """You are a chemistry specialist. Answer the question fully, strictly based on the provided documents.
Cite sources only at the end in the format: "Source: Document Title, Author".
Do not insert sources after each sentence.
If the answer is not in the documents, say: "I cannot find the answer in the provided documents".
Be accurate and informative."""
        }
        return prompts.get(lang_code, prompts['ru'])
    
    def split_into_chunks(self, text: str, title: str, author: str) -> List[Tuple[str, Dict]]:
        """Разделяет текст на семантические чанки"""
        chunks = []
        
        # Разделение на абзацы
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_chunk_paragraphs = []
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
                current_chunk_paragraphs.append(paragraph)
            else:
                if current_chunk:
                    metadata = {
                        'title': title,
                        'author': author,
                        'paragraph_count': len(current_chunk_paragraphs)
                    }
                    chunks.append((current_chunk.strip(), metadata))
                
                current_chunk = paragraph + "\n\n"
                current_chunk_paragraphs = [paragraph]
        
        if current_chunk:
            metadata = {
                'title': title,
                'author': author,
                'paragraph_count': len(current_chunk_paragraphs)
            }
            chunks.append((current_chunk.strip(), metadata))
        
        if not chunks and text:
            metadata = {'title': title, 'author': author, 'paragraph_count': 1}
            chunks.append((text.strip(), metadata))
        
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Создает эмбеддинги для чанков"""
        if EMBEDDING_MODEL_AVAILABLE:
            embeddings = self.embedding_model.encode(chunks)
            return embeddings
        else:
            print("Используется упрощенный эмбеддинг (BoW)")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=100)
            embeddings = vectorizer.fit_transform(chunks).toarray()
            return embeddings
    
    def preload_and_process_documents(self):
        """Загружает и обрабатывает документы для RAG"""
        folder_path = "/data"
        if not os.path.exists(folder_path):
            print(f"Папка не найдена: {folder_path}")
            return
        
        print("Загружаю и обрабатываю PDF файлы...")
        all_chunks_list = []
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf'):
                try:
                    # Загрузка PDF
                    text, title, author = self.load_pdf_from_file(
                        os.path.join(folder_path, filename)
                    )
                    self.loaded_documents[filename] = (text, title, author)
                    
                    # Чанкование
                    chunks_with_metadata = self.split_into_chunks(text, title, author)
                    self.document_chunks[filename] = chunks_with_metadata
                    
                    # Извлечение текста чанков для эмбеддингов
                    chunk_texts = [chunk for chunk, _ in chunks_with_metadata]
                    
                    # Создание эмбеддингов
                    if chunk_texts:
                        embeddings = self.create_embeddings(chunk_texts)
                        self.chunk_embeddings[filename] = embeddings
                        
                        # Добавление в общие структуры
                        for i, (chunk_text, metadata) in enumerate(chunks_with_metadata):
                            self.all_chunks.append(chunk_text)
                            self.chunk_to_doc.append({
                                'filename': filename,
                                'title': metadata['title'],
                                'author': metadata['author'],
                                'chunk_index': i
                            })
                        
                        print(f"Обработан: {filename} | Чанков: {len(chunk_texts)}")
                    else:
                        print(f"Нет чанков в файле: {filename}")
                        
                except Exception as e:
                    print(f"Ошибка обработки {filename}: {e}")
        
        # Создаем общую матрицу эмбеддингов
        if self.all_chunks:
            print("Создаю эмбеддинги для всех чанков...")
            self.all_embeddings = self.create_embeddings(self.all_chunks)
            print(f"Всего загружено: {len(self.loaded_documents)} файлов, {len(self.all_chunks)} чанков")
        else:
            print("Нет чанков для обработки")
    
    def load_pdf_from_file(self, file_path: str) -> tuple[str, str, str]:
        """Загружает текст из PDF и пытается определить название и автора"""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if not text.strip():
                raise Exception("Не удалось извлечь текст из PDF")
            
            # Метаданные
            metadata = pdf_reader.metadata or {}
            title = metadata.get('/Title')
            author = metadata.get('/Author')
            
            # Если метаданные пустые, пробуем из первых 15 строк текста
            if not title or title.strip() == "":
                first_lines = [line.strip() for line in text.splitlines() if line.strip()][:15]
                title = max(first_lines, key=len) if first_lines else os.path.basename(file_path)
            if not author or author.strip() == "":
                first_lines = [line.strip() for line in text.splitlines() if line.strip()][:15]
                author = next((line for line in first_lines if any(k in line.lower() for k in ["автор", "by", "editor", "редактор"])), "Неизвестен")
            
            return text, title, author
    
    def search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
        """Ищет наиболее релевантные чанки для запроса"""
        if not self.all_chunks or self.all_embeddings is None:
            return []
        
        # Создаем эмбеддинг для запроса
        if EMBEDDING_MODEL_AVAILABLE:
            query_embedding = self.embedding_model.encode([query])
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=100)
            all_texts = self.all_chunks + [query]
            vectorizer.fit(all_texts)
            query_embedding = vectorizer.transform([query]).toarray()
        
        similarities = cosine_similarity(query_embedding, self.all_embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Формируем результат
        results = []
        for idx in top_indices:
            if idx < len(self.all_chunks):
                chunk_text = self.all_chunks[idx]
                doc_info = self.chunk_to_doc[idx]
                results.append((chunk_text, doc_info))
        
        return results
    
    def get_gigachat_response(self, question: str, context_chunks: List[Tuple[str, Dict]], lang_code: str) -> str:
        """Получает ответ от GigaChat на основе релевантных чанков"""
        try:
            giga = GigaChat(
                credentials=self.gigachat_token,
                scope="GIGACHAT_API_PERS",
                model="GigaChat-2",
                verify_ssl_certs=False
            )
            
            system_prompt = self.get_system_prompt(lang_code)
            language_instruction = self.get_language_instruction(lang_code)
            
            # Формируем контекст из найденных чанков
            context_parts = []
            for chunk_text, doc_info in context_chunks:
                context_parts.append(f"Document: {doc_info['title']} by {doc_info['author']}\n{chunk_text}")
            
            context_text = "\n\n---\n\n".join(context_parts)
            
            # Ограничиваем размер контекста
            context_snippet = context_text[:6000]
            
            user_message = f"{question}\n\nRelevant documents:\n{context_snippet}\n\n{language_instruction}"
            
            payload = Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=system_prompt),
                    Messages(role=MessagesRole.USER, content=user_message)
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response = giga.chat(payload)
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Ошибка при обращении к GigaChat: {str(e)}"
    
    def register_handlers(self):
        self.dp.message(Command("start"))(self.cmd_start)
        self.dp.message(Command("list"))(self.cmd_list)
        self.dp.message(Command("help"))(self.cmd_help)
        self.dp.message(Command("stats"))(self.cmd_stats)
        self.dp.message(F.text)(self.handle_text_message)
    
    async def cmd_start(self, message: Message):
        files_count = len(self.loaded_documents)
        chunks_count = len(self.all_chunks)
        welcome_text = f"""
<b>Multi-Language Chemistry RAG Bot</b>

Статистика:
• Загружено файлов: {files_count}
• Обработано чанков: {chunks_count}
• Используется RAG с семантическим поиском

Просто задайте вопрос, и бот найдет релевантные фрагменты в документах!
"""
        await message.answer(welcome_text)
    
    async def cmd_help(self, message: Message):
        help_text = """
<b>Помощь / Help</b>

Команды:
/start - начать работу
/list - показать файлы
/stats - статистика обработки
/help - справка

Технология:
• RAG (Retrieval-Augmented Generation)
• Семантический поиск по эмбеддингам
• Чанкование документов

Просто задайте вопрос о химии или роботизации!
"""
        await message.answer(help_text)
    
    async def cmd_list(self, message: Message):
        if not self.loaded_documents:
            await message.answer("📭 Нет загруженных файлов")
            return
        
        files_info = []
        for filename, (text, title, author) in self.loaded_documents.items():
            chunks_count = len(self.document_chunks.get(filename, []))
            files_info.append(f"{filename}\n   Title: {title}\n   Author: {author}\n   Chunks: {chunks_count}")
        
        await message.answer(f"<b>Загруженные файлы:</b>\n\n" + "\n\n".join(files_info))
    
    async def cmd_stats(self, message: Message):
        stats_text = f"""
<b>Статистика RAG-системы</b>

• Загружено документов: {len(self.loaded_documents)}
• Всего чанков: {len(self.all_chunks)}
• Средний размер чанка: {self.chunk_size} символов
• Используется эмбеддинг: {'sentence-transformers' if EMBEDDING_MODEL_AVAILABLE else 'упрощенный TF-IDF'}

Чанков по документам:"""
        
        for filename, chunks in self.document_chunks.items():
            stats_text += f"\n• {filename}: {len(chunks)} чанков"
        
        await message.answer(stats_text)
    
    async def handle_text_message(self, message: Message):
        question = message.text.strip()
        if not self.loaded_documents:
            await message.answer("📭 Нет загруженных документов")
            return
        
        # Определяем язык
        lang_code = self.detect_language(question)
        
        # Информируем пользователя о процессе
        processing_msg = await message.answer("<b>Ищу релевантные фрагменты в документах...</b>")
        
        # Поиск релевантных чанков с помощью RAG
        relevant_chunks = self.search_relevant_chunks(question, top_k=5)
        
        if not relevant_chunks:
            await processing_msg.edit_text("Не найдено релевантной информации в документах.")
            return
        
        # Отправляем запрос в GigaChat с найденными чанками
        response = self.get_gigachat_response(question, relevant_chunks, lang_code)
        
        # Форматируем ответ
        sources_info = "\n".join([f"• {doc_info['title']} by {doc_info['author']}" 
                                  for _, doc_info in relevant_chunks])
        
        formatted_response = f"""
<b>Вопрос:</b> {question}

<b>Ответ:</b>
{response}

<b>Использованные источники:</b>
{sources_info}

<i>Ответ сгенерирован с использованием RAG-системы</i>
"""
        
        await processing_msg.edit_text(formatted_response)
    
    async def run(self):
        print(f"Бот запущен. Загружено файлов: {len(self.loaded_documents)}, чанков: {len(self.all_chunks)}")
        await self.dp.start_polling(self.bot)


# Настройка
GIGACHAT_TOKEN = ""
TELEGRAM_TOKEN = ""

async def main():
    if not EMBEDDING_MODEL_AVAILABLE:
        print("Для лучшей работы установите: pip install sentence-transformers")
    
    bot = MultiLanguageGigaChatBot(GIGACHAT_TOKEN, TELEGRAM_TOKEN, chunk_size=500)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
