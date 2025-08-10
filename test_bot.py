import os
from telegram import Bot
from telegram.ext import ApplicationBuilder

# Load from Railway env vars
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def main():
    if not TOKEN or not CHAT_ID:
        print("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment.")
        return

    bot = Bot(token=TOKEN)

    try:
        await bot.send_message(chat_id=CHAT_ID, text="✅ Hello from Railway test bot!")
        print(f"✅ Message sent to chat ID {CHAT_ID}")
    except Exception as e:
        print(f"❌ Failed to send message: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
