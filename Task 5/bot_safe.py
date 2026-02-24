import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from rag_safe import ask_rag  # используем защищённую версию

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ===== ВСТАВЬТЕ СВОЙ ТОКЕН =====
BOT_TOKEN = ""
# =================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я RAG-бот по базе знаний (модифицированная вселенная Star Wars). Задай мне вопрос!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    logger.info(f"Вопрос: {question}")
    await update.message.chat.send_action(action="typing")
    answer = ask_rag(question)
    await update.message.reply_text(answer)

def main():
    if BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("ОШИБКА: Замените BOT_TOKEN на реальный токен в файле bot_safe.py")
        return
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()