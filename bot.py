import os
import pypdf
import asyncio
import requests
import json
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
        
        # Загружаем файлы
        self.preload_documents()
        
        # Инициализация aiogram
        self.bot = Bot(
            token=telegram_token, 
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        self.dp = Dispatcher()
        
        # Регистрация обработчиков
        self.register_handlers()

    def detect_language(self, text: str) -> str:
        """Определяет язык текста и возвращает код языка"""
        try:
            if detect is None:
                return 'ru'
            lang_code = detect(text)
            return lang_code
        except:
            return 'ru'
    
    def get_language_name(self, lang_code: str) -> str:
        """Возвращает название языка по коду"""
        language_names = {
            'ru': 'русском',
            'en': 'английском', 
            'de': 'немецком',
            'fr': 'французском',
            'es': 'испанском',
            'it': 'итальянском',
            'zh': 'китайском',
            'ja': 'японском',
            'ko': 'корейском',
            'ar': 'арабском',
            'pt': 'португальском',
            'uk': 'украинском',
            'pl': 'польском'
        }
        return language_names.get(lang_code, 'русском')
    
    def get_language_instruction(self, lang_code: str) -> str:
        """Возвращает инструкцию для AI на нужном языке"""
        instructions = {
            'ru': "Отвечай строго на русском языке.",
            'en': "Answer strictly in English.",
            'de': "Antworte streng auf Deutsch.",
            'fr': "Réponds strictement en français.",
            'es': "Responde estrictamente en español.",
            'it': "Rispondi rigorosamente in italiano.",
            'zh': "请严格用中文回答。",
            'ja': "厳密に日本語で答えてください。",
            'ko': "엄격하게 한국어로 답변해 주세요.",
            'ar': "الرد بدقة باللغة العربية.",
            'pt': "Responda estritamente em português.",
            'uk': "Відповідайте строго українською мовою.",
            'pl': "Odpowiadaj ściśle po polsku."
        }
        return instructions.get(lang_code, "Отвечай строго на русском языке.")
    
    def get_system_prompt(self, lang_code: str) -> str:
        """Возвращает системный промпт на нужном языке"""
        prompts = {
            'ru': """Вы специалист по химии. Отвечайте строго на основе предоставленных фрагментов документа. 
Соблюдайте эти правила:
1. Отвечайте только на основании предоставленной информации
2. Если информации недостаточно для получения полного ответа - укажите это
3. Если ответа нет в документах - скажите: "Я не могу найти ответ в предоставленных документах"
4. Будьте точны и информативны
5. Правильно используйте химическую терминологию""",
            'en': """You are a chemistry specialist. Answer strictly based on the provided document fragments.
Follow these rules:
1. Answer only based on the provided information
2. If there is not enough information for a complete answer - indicate this
3. If the answer is not in the documents - say: "I cannot find the answer in the provided documents"
4. Be accurate and informative
5. Use chemical terminology correctly""",
            'de': """Sie sind ein Chemiespezialist. Antworten Sie streng auf der Grundlage der bereitgestellten Dokumentenfragmente.
Befolgen Sie diese Regeln:
1. Antworten Sie nur auf der Grundlage der bereitgestellten Informationen
2. Wenn nicht genügend Informationen für eine vollständige Antwort vorhanden sind - weisen Sie darauf hin
3. Wenn die Antwort nicht in den Dokumenten steht - sagen Sie: "Ich kann die Antwort in den bereitgestellten Dokumenten nicht finden"
4. Seien Sie genau und informativ
5. Verwenden Sie die chemische Terminologie korrekt""",
            'fr': """Vous êtes un spécialiste de la chimie. Répondez strictement sur la base des fragments de documents fournis.
Suivez ces règles :
1. Répondez uniquement sur la base des informations fournies
2. S'il n'y a pas assez d'informations pour une réponse complète - indiquez-le
3. Si la réponse n'est pas dans les documents - dites : "Je ne peux pas trouver la réponse dans les documents fournis"
4. Soyez précis et informatif
5. Utilisez correctement la terminologie chimique"""
        }
        return prompts.get(lang_code, prompts['ru'])
    
    def get_processing_message(self, lang_code: str) -> str:
        """Возвращает сообщение о обработке на нужном языке"""
        messages = {
            'ru': "<b>🤔 Анализирую документы...</b>",
            'en': "<b>🤔 Analyzing documents...</b>",
            'de': "<b>🤔 Dokumente werden analysiert...</b>",
            'fr': "<b>🤔 Analyse des documents...</b>",
            'es': "<b>🤔 Analizando documentos...</b>",
            'it': "<b>🤔 Analizzando documenti...</b>"
        }
        return messages.get(lang_code, messages['ru'])
    
    def get_no_documents_message(self, lang_code: str) -> str:
        """Возвращает сообщение об отсутствии документов на нужном языке"""
        messages = {
            'ru': "📭 Нет загруженных документов",
            'en': "📭 No documents loaded",
            'de': "📭 Keine Dokumente geladen",
            'fr': "📭 Aucun document chargé",
            'es': "📭 No hay documentos cargados",
            'it': "📭 Nessun documento caricato"
        }
        return messages.get(lang_code, messages['ru'])
    
    def validate_tokens(self):
        """Проверяет валидность токенов"""
        if self.gigachat_token == "0" or self.telegram_token == "0":
            print("❌ ОШИБКА: Замените токены на реальные!")
            print("📝 Как получить токены:")
            print("1. GigaChat: https://developers.sber.ru/studio/products/gigachatapi")
            print("2. Telegram: напишите @BotFather -> /newbot")
            return False
        return True
    
    def preload_documents(self):
        """Загружает все PDF файлы из папки"""
        folder_path = "/Users/kseniatebenkova/Desktop/data"  

        if not os.path.exists(folder_path):
            print(f"⚠️ Папка не найдена: {folder_path}")
            return

        print("📥 Загружаю PDF файлы...")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(folder_path, filename)
                try:
                    text = self.load_pdf_from_file(file_path)
                    self.loaded_documents[filename] = text
                    print(f"✅ Загружен: {filename} ({len(text)} символов)")
                except Exception as e:
                    print(f"❌ Ошибка загрузки {filename}: {e}")
        
        print(f"📚 Всего загружено: {len(self.loaded_documents)} файлов")
    
    def load_pdf_from_file(self, file_path: str) -> str:
        """Загружает текст из PDF файла"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if not text.strip():
                    raise Exception("Не удалось извлечь текст из PDF")
                
                return text
        except Exception as e:
            raise Exception(f"Ошибка загрузки PDF: {str(e)}")
    
    def get_gigachat_response(self, question: str, context: str = "", lang_code: str = "ru") -> str:
        """Получает ответ от GigaChat на нужном языке"""
        try:
            if GigaChat is None:
                return "❌ Библиотека GigaChat не установлена"

            giga = GigaChat(
                credentials=self.gigachat_token,
                scope="GIGACHAT_API_PERS",
                model="GigaChat-2",
                verify_ssl_certs=False
            )

            system_prompt = self.get_system_prompt(lang_code)
            language_instruction = self.get_language_instruction(lang_code)

            # Формируем промпт на нужном языке
            if lang_code == 'ru':
                user_message = f"""Вопрос: {question}

Контекст:
{context[:6000]}

Основываясь на предоставленных документах, дайте точный ответ на вопрос. Если информации недостаточно, укажите это. {language_instruction}"""
            elif lang_code == 'en':
                user_message = f"""Question: {question}

Context:
{context[:6000]}

Based on the provided documents, give an accurate answer to the question. If there is not enough information, indicate this. {language_instruction}"""
            elif lang_code == 'de':
                user_message = f"""Frage: {question}

Kontext:
{context[:6000]}

Geben Sie auf der Grundlage der bereitgestellten Dokumente eine genaue Antwort auf die Frage. Wenn nicht genügend Informationen vorhanden sind, weisen Sie darauf hin. {language_instruction}"""
            elif lang_code == 'fr':
                user_message = f"""Question: {question}

Contexte:
{context[:6000]}

Sur la base des documents fournis, donnez une réponse précise à la question. S'il n'y a pas assez d'informations, indiquez-le. {language_instruction}"""
            else:
                # Для других языков используем английский как fallback
                user_message = f"""Question: {question}

Context:
{context[:6000]}

Based on the provided documents, give an accurate answer to the question. If there is not enough information, indicate this. {language_instruction}"""
            
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
            error_messages = {
                'ru': f"❌ Ошибка при обращении к GigaChat: {str(e)}",
                'en': f"❌ Error accessing GigaChat: {str(e)}",
                'de': f"❌ Fehler beim Zugriff auf GigaChat: {str(e)}",
                'fr': f"❌ Erreur d'accès à GigaChat: {str(e)}"
            }
            return error_messages.get(lang_code, error_messages['ru'])
    
    def register_handlers(self):
        """Регистрирует обработчики"""
        self.dp.message(Command("start"))(self.cmd_start)
        self.dp.message(Command("list"))(self.cmd_list)
        self.dp.message(Command("help"))(self.cmd_help)
        self.dp.message(F.text)(self.handle_text_message)
    
    async def cmd_start(self, message: Message):
        """Обработчик команды /start"""
        files_count = len(self.loaded_documents)
        
        welcome_text = f"""
<b>🤖 Multi-Language Chemistry PDF Bot</b>
<b>🌍 Многоязычный Химический PDF Бот</b>

📚 <b>Загружено файлов / Files loaded:</b> {files_count}

<b>Поддерживаемые языки / Supported languages:</b>
• 🇷🇺 Русский / Russian
• 🇺🇸 English / Английский  
• 🇩🇪 Deutsch / Немецкий
• 🇫🇷 Français / Французский
• 🇪🇸 Español / Испанский
• 🇮🇹 Italiano / Итальянский
• и другие / and others

<b>Просто задайте вопрос на любом языке!</b>
<b>Just ask a question in any language!</b>
"""
        await message.answer(welcome_text)
    
    async def cmd_help(self, message: Message):
        """Показывает справку"""
        help_text = """
<b>📖 Помощь / Help</b>

<b>Команды / Commands:</b>
/start - начать работу / start bot
/list - показать файлы / show files  
/help - справка / help

<b>Просто задайте вопрос на любом языке о химических документах!</b>
<b>Just ask any question in any language about chemistry documents!</b>
"""
        await message.answer(help_text)
    
    async def cmd_list(self, message: Message):
        """Показывает загруженные файлы"""
        if not self.loaded_documents:
            await message.answer("📭 Нет загруженных файлов / No files loaded")
            return
        
        files_info = []
        for filename, text in self.loaded_documents.items():
            files_info.append(f"📄 {filename} - {len(text)} chars")
        
        response = "<b>📚 Загруженные файлы / Loaded files:</b>\n" + "\n".join(files_info)
        await message.answer(response)
    
    async def handle_text_message(self, message: Message):
        """Обработчик вопросов на любом языке"""
        user_message = message.text.strip()
        
        if not self.loaded_documents:
            # Определяем язык вопроса для сообщения об ошибке
            lang_code = self.detect_language(user_message)
            error_msg = self.get_no_documents_message(lang_code)
            await message.answer(error_msg)
            return
        
        # Определяем язык вопроса
        lang_code = self.detect_language(user_message)
        lang_name = self.get_language_name(lang_code)
        
        print(f"🌐 Определен язык: {lang_name} ({lang_code})")
        
        # Получаем сообщение о обработке на нужном языке
        processing_msg_text = self.get_processing_message(lang_code)
        processing_msg = await message.answer(processing_msg_text)
        
        try:
            # Объединяем все документы в контекст
            context_text = ""
            for filename, text in self.loaded_documents.items():
                context_text += f"--- {filename} ---\n{text}\n\n"
            
            # Получаем ответ от GigaChat на нужном языке
            response = self.get_gigachat_response(user_message, context_text, lang_code)
            
            # Форматируем ответ с метками на нужном языке
            question_labels = {
                'ru': "<b>Вопрос:</b>",
                'en': "<b>Question:</b>", 
                'de': "<b>Frage:</b>",
                'fr': "<b>Question:</b>",
                'es': "<b>Pregunta:</b>",
                'it': "<b>Domanda:</b>"
            }
            
            answer_labels = {
                'ru': "<b>Ответ:</b>",
                'en': "<b>Answer:</b>",
                'de': "<b>Antwort:</b>", 
                'fr': "<b>Réponse:</b>",
                'es': "<b>Respuesta:</b>",
                'it': "<b>Risposta:</b>"
            }
            
            question_label = question_labels.get(lang_code, "<b>Question:</b>")
            answer_label = answer_labels.get(lang_code, "<b>Answer:</b>")
            
            formatted_response = f"{question_label} {user_message}\n\n{answer_label}\n{response}"
            await processing_msg.edit_text(formatted_response)
            
        except Exception as e:
            error_messages = {
                'ru': f"❌ <b>Ошибка:</b> {str(e)}",
                'en': f"❌ <b>Error:</b> {str(e)}",
                'de': f"❌ <b>Fehler:</b> {str(e)}",
                'fr': f"❌ <b>Erreur:</b> {str(e)}"
            }
            error_msg = error_messages.get(lang_code, error_messages['ru'])
            await processing_msg.edit_text(error_msg)
    
    async def run(self):
        """Запускает бота"""
        print("=" * 60)
        print(f"🤖 Многоязычный химический бот запущен!")
        print(f"📚 Загружено файлов: {len(self.loaded_documents)}")
        print("🌍 Поддерживаемые языки: RU, EN, DE, FR, ES, IT, ZH, JA, KO, AR, PT, UK, PL")
        for filename in self.loaded_documents.keys():
            print(f"📄 {filename}")
        print("=" * 60)
        await self.dp.start_polling(self.bot)

# Настройка
GIGACHAT_TOKEN = ""
TELEGRAM_TOKEN = ""

# Запуск
async def main():
    bot = MultiLanguageGigaChatBot(GIGACHAT_TOKEN, TELEGRAM_TOKEN)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())