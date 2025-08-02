import json
import os
import asyncio
import logging
from typing import Set, Optional

from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from openai import AsyncOpenAI
import dotenv

dotenv.load_dotenv()

# Logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("data/bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a helpful AI assistant for Telegram. Your users are college students.  
- Always answer in Markdown format.  
- Limit responses to a maximum of two concise paragraphs.  
- Use relevant emojis to make answers engaging, if possible.  
- Keep your tone friendly, clear, and supportive.
"""


class Storage:
    def __init__(self, vip_path="data/vip_users.json", offset_path="data/last_offset.json", memory_path="data/memory.json"):
        self.vip_path = vip_path
        self.offset_path = offset_path
        self.memory_path = memory_path

        os.makedirs("data", exist_ok=True)

    async def load_vip_users(self) -> Set[int]:
        try:
            with open(self.vip_path, "r", encoding="utf8") as f:
                data = json.load(f)
                return set(data.get("vip_users", []))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    async def save_vip_users(self, vip_users: Set[int]):
        with open(self.vip_path, "w", encoding="utf8") as f:
            json.dump({"vip_users": list(vip_users)}, f, indent=2)

    async def load_offset(self) -> Optional[int]:
        try:
            with open(self.offset_path, "r", encoding="utf8") as f:
                return json.load(f).get("offset")
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    async def save_offset(self, offset: int):
        with open(self.offset_path, "w", encoding="utf8") as f:
            json.dump({"offset": offset}, f, indent=2)

    def load_memory(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_memory(self):
        with open(self.memory_path, "w", encoding="utf8") as f:
            json.dump(self._data, f, indent=2)


class Memory:
    def __init__(self, storage, memory_path="data/memory.json"):
        self.memory_path = memory_path
        self.storage = storage
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        self._data = self.storage.load_memory()

    def add_message(self, chat_id: int, role: str, message: str):
        chat_id_str = str(chat_id)
        if chat_id_str not in self._data:
            self._data[chat_id_str] = []
        self._data[chat_id_str].append({"role": role, "message": message})
        # Keep last 100
        self._data[chat_id_str] = self._data[chat_id_str][-100:]
        self.storage.save_memory()

    def get_last_messages(self, chat_id: int, limit=10):
        return self._data.get(str(chat_id), [])[-limit:]


class OpenAIClient:
    def __init__(self, api_key: str, base_url: str, model: str, max_tokens: int, temperature: float):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def get_response(self, user_text: str, history: list[dict]) -> str:
        try:

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

            messages += [{"role": msg["role"], "content": msg["message"]}
                         for msg in history]

            messages.append({"role": "user", "content": user_text})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"❌ Error: {e}"

    async def close(self):
        await self.client.close()


class TelegramBot:
    def __init__(self, token: str, openai_client: OpenAIClient, storage: Storage,
                 memory: Memory, vip_code: str, limit: int, timeout: int):
        self.bot = Bot(token=token)
        self.openai = openai_client
        self.storage = storage
        self.memory = memory
        self.vip_code = vip_code
        self.limit = limit
        self.timeout = timeout
        self.vip_users: Set[int] = set()
        self.offset: Optional[int] = None

    async def start(self):
        self.offset = await self.storage.load_offset()
        self.vip_users = await self.storage.load_vip_users()
        logger.info("🤖 Bot is now running...")

        while True:
            try:
                updates = await self.bot.get_updates(
                    offset=self.offset,
                    limit=self.limit,
                    timeout=self.timeout,
                    allowed_updates=["message"]
                )
                if updates:
                    await asyncio.gather(*[self.process_update(update) for update in updates])
                    self.offset = updates[-1].update_id + 1
                    await self.storage.save_offset(self.offset)
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error fetching updates: {e}")
                await asyncio.sleep(10)

    async def process_update(self, update: Update):
        message = update.message
        if not message or not message.text:
            return

        chat_id = message.chat.id
        text = message.text.strip()

        if text == self.vip_code:
            self.vip_users.add(chat_id)
            await self.storage.save_vip_users(self.vip_users)
            await self.send_message(chat_id, "🎉 You are now a VIP user!")
            return

        if chat_id not in self.vip_users:
            await self.send_message(chat_id, "Only VIP users can use this bot.")
            return

        try:
            await self.bot.send_chat_action(chat_id, "typing")
            history = self.memory.get_last_messages(chat_id, limit=10)
            reply = await self.openai.get_response(text, history)
            await self.send_message(chat_id, reply, reply_to=message.message_id)

            self.memory.add_message(chat_id, "user", text)
            self.memory.add_message(chat_id, "assistant", reply)

        except Exception as e:
            logger.error(f"Failed to process message: {e}")

    async def send_message(self, chat_id: int, text: str, reply_to: Optional[int] = None):
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_to_message_id=reply_to
            )
        except TelegramError:
            await self.bot.send_message(chat_id=chat_id, text=text, reply_to_message_id=reply_to)

    async def shutdown(self):
        logger.info("🛑 Shutting down bot...")
        await self.openai.close()


async def main():
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    VIP_CODE = os.getenv("VIP_CODE", "VIP123")

    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 500))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
    LIMIT = int(os.getenv("LIMIT", 10))
    TIMEOUT = int(os.getenv("TIMEOUT", 30))

    if not BOT_TOKEN or not OPENAI_API_KEY or not OPENAI_API_BASE:
        logger.critical("Missing required environment variables.")
        return

    storage = Storage()
    memory = Memory(storage)
    openai_client = OpenAIClient(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    telegram_bot = TelegramBot(
        BOT_TOKEN, openai_client, storage, memory, VIP_CODE, LIMIT, TIMEOUT)

    try:
        await telegram_bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        await telegram_bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
