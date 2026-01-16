import os
import pypdf
import asyncio

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

class MultiLanguageGigaChatBot:
    def __init__(self, gigachat_token: str, telegram_token: str):
        self.gigachat_token = gigachat_token
        self.telegram_token = telegram_token
        self.loaded_documents = {}
        
        self.preload_documents()
        
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
Будьте точны и информативны.""",
            'en': """You are a chemistry specialist. Answer the question fully, strictly based on the provided documents.
Cite sources only at the end in the format: "Source: Document Title, Author".
Do not insert sources after each sentence.
If the answer is not in the documents, say: "I cannot find the answer in the provided documents".
Be accurate and informative."""
        }
        return prompts.get(lang_code, prompts['ru'])
    
    def preload_documents(self):
        folder_path = "/data"
        if not os.path.exists(folder_path):
            print(f"Папка не найдена: {folder_path}")
            return
        print("Загружаю PDF файлы...")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf'):
                try:
                    text, title, author = self.load_pdf_from_file(os.path.join(folder_path, filename))
                    self.loaded_documents[filename] = (text, title, author)
                    print(f"Загружен: {filename} | Title: {title} | Author: {author}")
                except Exception as e:
                    print(f"Ошибка загрузки {filename}: {e}")
        print(f"Всего загружено: {len(self.loaded_documents)} файлов")
    
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
    
    def get_gigachat_response(self, question: str, context: str, lang_code: str) -> str:
        try:
            giga = GigaChat(
                credentials=self.gigachat_token,
                scope="GIGACHAT_API_PERS",
                model="GigaChat-2",
                verify_ssl_certs=False
            )
            system_prompt = self.get_system_prompt(lang_code)
            language_instruction = self.get_language_instruction(lang_code)
            
            context_snippet = context[:6000]
            user_message = f"{question}\n\nContext:\n{context_snippet}\n\n{language_instruction}"
            
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
        self.dp.message(F.text)(self.handle_text_message)
    
    async def cmd_start(self, message: Message):
        files_count = len(self.loaded_documents)
        welcome_text = f"""
<b>Multi-Language Chemistry PDF Bot</b>
Загружено файлов: {files_count}
Просто задайте вопрос, и бот ответит с полным ответом и источником в конце!
"""
        await message.answer(welcome_text)
    
    async def cmd_help(self, message: Message):
        help_text = """
<b>Помощь / Help</b>
Команды:
/start - начать работу
/list - показать файлы
/help - справка
Просто задайте вопрос о химии или роботизации, и бот ответит с источником в конце!
"""
        await message.answer(help_text)
    
    async def cmd_list(self, message: Message):
        if not self.loaded_documents:
            await message.answer("Нет загруженных файлов")
            return
        files_info = "\n".join([f"{fn} | Title: {title} | Author: {author}" 
                                for fn, (text, title, author) in self.loaded_documents.items()])
        await message.answer(f"<b>Загруженные файлы:</b>\n{files_info}")
    
    async def handle_text_message(self, message: Message):
        question = message.text.strip()
        if not self.loaded_documents:
            await message.answer("📭 Нет загруженных документов")
            return

        lang_code = self.detect_language(question)
        processing_msg = await message.answer("<b>🤔 Анализирую документы...</b>")

        # Формируем отдельные блоки для каждого документа с метаданными
        context_messages = []
        for filename, (text, title, author) in self.loaded_documents.items():
            snippet = text[:2000]  # Ограничение, чтобы не перегружать
            context_messages.append(f"Document: \"{title}\" by {author}\n{snippet}")

        # Объединяем все блоки в один контекст
        context_text = "\n\n".join(context_messages)

        # Добавляем системный промпт и инструкцию
        system_prompt = self.get_system_prompt(lang_code)
        language_instruction = self.get_language_instruction(lang_code)

        # Формируем сообщение для GigaChat
        user_message = f"{question}\n\nContext:\n{context_text}\n\n{language_instruction}\n\n" \
                      "Cite all relevant documents in the format: \"Source: Document Title, Author\". " \
                      "If no document contains the answer, do not provide any source."

        # Отправка запроса в GigaChat
        response = self.get_gigachat_response(question, context_text, lang_code)

        formatted_response = f"<b>Вопрос:</b> {question}\n\n<b>Ответ:</b>\n{response}"
        await processing_msg.edit_text(formatted_response)
    
    async def run(self):
        print(f"Бот запущен. Загружено файлов: {len(self.loaded_documents)}")
        for fn, (_, title, author) in self.loaded_documents.items():
            print(f"{fn} | Title: {title} | Author: {author}")
        await self.dp.start_polling(self.bot)

# Настройка
GIGACHAT_TOKEN = "_token_"
TELEGRAM_TOKEN = "_token_"

async def main():
    bot = MultiLanguageGigaChatBot(GIGACHAT_TOKEN, TELEGRAM_TOKEN)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
