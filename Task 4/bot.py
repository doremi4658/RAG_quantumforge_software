import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from rag import ask_rag

# Включим логирование
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен бота (вставьте свой)
BOT_TOKEN = ""

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(
        "Привет! Я RAG-бот по базе знаний (модифицированная вселенная Star Wars). Задай мне вопрос!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    question = update.message.text
    logger.info(f"Вопрос: {question}")

    # Отправляем "печатает..."
    await update.message.chat.send_action(action="typing")

    # Получаем ответ от RAG
    answer = ask_rag(question)

    # Отправляем ответ
    await update.message.reply_text(answer)

def main():
    """Запуск бота"""
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()