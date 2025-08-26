import os
import json
import asyncio
import logging
from typing import Optional, Set, Dict, Any, List

from telegram import Bot, Update
from telegram.constants import ParseMode, ChatAction
from telegram.error import TelegramError

from openai import AsyncOpenAI
import dotenv

dotenv.load_dotenv()

# =========================
# Logging
# =========================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler("data/bot.log")]
)
logger = logging.getLogger("tg-assistants-bot")

# =========================
# Config & System Prompt
# =========================
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Telegram Ai Assiatnt")
ASSISTANT_DESCRIPTION = "A friendly AI assistant for college students on Telegram."
# FIX: Changed model to a real, powerful model supporting vision & tools. 'gpt-4.1' does not exist.
ASSISTANT_MODEL = os.getenv("MODEL_NAME", "gpt-4.1")
# Built-in tools: code_interpreter, file_search, and vision via image_file content.
ASSISTANT_TOOLS = [
    {"type": "code_interpreter"},
    {"type": "file_search"},
    # {"type": "web_search"},       # NEW: OpenAI built-in web search
    # {"type": "vision"}            # NEW: Vision tool (for image understanding)
]


SYSTEM_INSTRUCTIONS = """
You are a helpful AI assistant for Telegram. Your users are college students.
- Always answer in **Markdown**.
- Limit responses to **a maximum of two concise paragraphs** if not otherwise requested.
- Use relevant emojis to make answers engaging.
- Keep your tone friendly, clear, and supportive.
- When users send images, analyze them using the vision tool.
- When users send PDFs or documents, summarize and answer based on contents.
- Use web search when you need up-to-date factual answers.
- Prefer short, well-structured answers with bullet points if that helps clarity.
"""

# =========================
# Storage Layer
# =========================


class Storage:
    """
    Persists:
      - VIP users
      - last update offset
      - chat threads mapping
      - assistant id cache
    """

    def __init__(
        self,
        base_dir: str = "data",
        vip_path: str = "data/vip_users.json",
        offset_path: str = "data/last_offset.json",
        threads_path: str = "data/threads.json",
        assistant_path: str = "data/assistant.json",
    ):
        self.base_dir = base_dir
        self.vip_path = vip_path
        self.offset_path = offset_path
        self.threads_path = threads_path
        self.assistant_path = assistant_path

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "files"), exist_ok=True)

    async def load_vip_users(self) -> Set[int]:
        try:
            with open(self.vip_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("vip_users", []))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    async def save_vip_users(self, vip_users: Set[int]):
        with open(self.vip_path, "w", encoding="utf-8") as f:
            json.dump({"vip_users": list(vip_users)}, f, indent=2)

    async def load_offset(self) -> Optional[int]:
        try:
            with open(self.offset_path, "r", encoding="utf-8") as f:
                return json.load(f).get("offset")
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    async def save_offset(self, offset: int):
        with open(self.offset_path, "w", encoding="utf-8") as f:
            json.dump({"offset": offset}, f, indent=2)

    async def load_threads(self) -> Dict[str, str]:
        try:
            with open(self.threads_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    async def save_threads(self, mapping: Dict[str, str]):
        with open(self.threads_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

    async def load_assistant_id(self) -> Optional[str]:
        try:
            with open(self.assistant_path, "r", encoding="utf-8") as f:
                return json.load(f).get("assistant_id")
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    async def save_assistant_id(self, assistant_id: str):
        with open(self.assistant_path, "w", encoding="utf-8") as f:
            json.dump({"assistant_id": assistant_id}, f, indent=2)

# =========================
# OpenAI Assistants Client
# =========================


class AssistantsClient:
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.assistant_id: Optional[str] = None

    async def ensure_assistant(self) -> str:
        """
        Create the assistant once (id cached by caller). Idempotent enough for our use.
        """
        # The caller is responsible for caching this ID; here we only create.
        assistant = await self.client.beta.assistants.create(
            name=ASSISTANT_NAME,
            description=ASSISTANT_DESCRIPTION,
            model=ASSISTANT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            tools=ASSISTANT_TOOLS,
        )
        self.assistant_id = assistant.id
        logger.info(f"Assistant ready: {assistant.id}")
        return assistant.id

    async def create_thread(self) -> str:
        thread = await self.client.beta.threads.create()
        return thread.id

    async def upload_file(self, path: str) -> str:
        """
        Upload a file for use with Assistants (purpose='assistants').
        Returns file_id.
        """
        with open(path, "rb") as f:
            uploaded = await self.client.files.create(
                file=f,
                purpose="assistants",
            )
        return uploaded.id

    async def send_user_message_text(
        self, thread_id: str, text: str, attachments: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Post a user message with text (and optional attachments for file_search).
        """
        # FIX: Simplified the message creation call. The 'content' for a simple
        # text message is just the text string. Attachments are passed separately.
        await self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=text,
            attachments=attachments or []
        )

    async def send_user_message_with_images(
        self, thread_id: str, prompt: str, image_file_ids: List[str]
    ):
        """
        Post a user message with one or more images using a multi-part content body.
        """
        # FIX: Changed content part types from 'input_text' and 'input_image' to
        # the correct 'text' and 'image_file' types.
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt or ""}]

        for fid in image_file_ids:
            content.append(
                {"type": "image_file", "image_file": {"file_id": fid}})
        await self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content,
        )

    async def run_and_wait(self, thread_id: str, assistant_id: str, poll_interval: float = 0.8, timeout: float = 120.0):
        """
        Create a run and poll until completed/failed/expired/cancelled.
        """
        run = await self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

        # Poll
        elapsed = 0.0
        while True:
            current = await self.client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            status = current.status
            if status in ("completed", "failed", "cancelled", "expired"):
                return current
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            if elapsed > timeout:
                raise TimeoutError("Run polling timed out.")

    async def get_last_assistant_message_text(self, thread_id: str) -> str:
        """
        Get the latest assistant message text in the thread.
        """
        msgs = await self.client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=10)
        for m in msgs.data:
            if m.role == "assistant":
                # Concatenate all text blocks
                parts = []
                for c in m.content:
                    # FIX: Simplified to check only for the 'text' content type, which is the
                    # standard for the current API. Kept defensive checks.
                    if c.type == "text" and hasattr(c, 'text') and hasattr(c.text, 'value'):
                        parts.append(c.text.value)
                if parts:
                    return "\n".join(parts).strip()
        return "I couldn't find a fresh reply—please try again."

    async def close(self):
        await self.client.close()

# =========================
# Telegram Bot
# =========================


class TelegramAssistantsBot:
    def __init__(
        self,
        bot_token: str,
        openai_client: AssistantsClient,
        storage: Storage,
        vip_code: str,
        limit: int,
        timeout: int,
    ):
        self.bot = Bot(token=bot_token)
        self.ai = openai_client
        self.storage = storage
        self.vip_code = vip_code
        self.limit = limit
        self.timeout = timeout

        self.vip_users: Set[int] = set()
        self.offset: Optional[int] = None
        self.threads: Dict[str, str] = {}  # chat_id(str) -> thread_id
        self.assistant_id: Optional[str] = None

    async def start(self):
        # Load persisted state
        self.offset = await self.storage.load_offset()
        self.vip_users = await self.storage.load_vip_users()
        self.threads = await self.storage.load_threads()

        # Ensure assistant exists, cache ID
        cached = await self.storage.load_assistant_id()
        if cached:
            self.assistant_id = cached
            logger.info(f"Using cached assistant: {self.assistant_id}")
        else:
            self.assistant_id = await self.ai.ensure_assistant()
            await self.storage.save_assistant_id(self.assistant_id)

        logger.info("🤖 Bot is now running with Assistants API...")

        # Main update loop
        while True:
            try:
                updates = await self.bot.get_updates(
                    offset=self.offset,
                    limit=self.limit,
                    timeout=self.timeout,
                    allowed_updates=["message"]
                )
                if updates:
                    await asyncio.gather(*[self.process_update(u) for u in updates])
                    self.offset = updates[-1].update_id + 1
                    await self.storage.save_offset(self.offset)
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error fetching updates: {e}")
                await asyncio.sleep(5)

    async def ensure_thread(self, chat_id: int) -> str:
        key = str(chat_id)
        if key in self.threads:
            return self.threads[key]
        t_id = await self.ai.create_thread()
        self.threads[key] = t_id
        await self.storage.save_threads(self.threads)
        logger.info(f"Created thread for chat {chat_id}: {t_id}")
        return t_id

    async def process_update(self, update: Update):
        message = update.message
        if not message:
            return

        chat_id = message.chat.id
        text = (message.text or "").strip()

        # VIP enrollment
        if text and text == self.vip_code:
            self.vip_users.add(chat_id)
            await self.storage.save_vip_users(self.vip_users)
            await self.safe_send(chat_id, "🎉 You are now a VIP user! Enjoy unlimited access.")
            return

        # Access control
        if chat_id not in self.vip_users:
            await self.safe_send(chat_id, "🔒 Only VIP users can use this bot. Send your VIP code to proceed.")
            return

        try:
            await self.bot.send_chat_action(chat_id, ChatAction.TYPING)
            thread_id = await self.ensure_thread(chat_id)

            if message.photo:
                await self.handle_photo(chat_id, message, thread_id)
            elif message.document:
                await self.handle_document(chat_id, message, thread_id)
            elif message.text:
                await self.handle_text(chat_id, text, thread_id)
            else:
                await self.safe_send(chat_id, "❌ Unsupported message type. Please send text, images, or documents.")
        except Exception as e:
            logger.exception("Failed to process update")
            await self.safe_send(chat_id, "❌ An error occurred while processing your message.")

    async def handle_text(self, chat_id: int, text: str, thread_id: str):
        # Plain text message to the thread
        await self.ai.send_user_message_text(thread_id, text)
        await self.run_and_reply(chat_id, thread_id)

    async def handle_photo(self, chat_id: int, message, thread_id: str):
        """
        Downloads the largest photo, uploads to OpenAI, sends as input_image
        to the thread, runs, and replies.
        """
        # FIX: Initialize local_path to prevent NameError in the finally block
        # if an exception occurs before its assignment.
        local_path = None
        try:
            photo = message.photo[-1]
            file_obj = await self.bot.get_file(photo.file_id)
            local_path = f"data/files/{photo.file_id}.jpg"
            await file_obj.download_to_drive(local_path)

            file_id = await self.ai.upload_file(local_path)
            caption = message.caption or "Please analyze this image."

            await self.ai.send_user_message_with_images(thread_id, caption, [file_id])
            await self.run_and_reply(chat_id, thread_id)

        except Exception as e:
            logger.error(f"Image handling error: {e}")
            await self.safe_send(chat_id, "❌ Failed to process the image.")
        finally:
            try:
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)
            except Exception as e:
                logger.error(f"Error removing temp image file: {e}")

    async def handle_document(self, chat_id: int, message, thread_id: str):
        """
        Handles documents (including PDFs) by uploading them for file_search tool.
        """
        document = message.document
        local_path = None
        try:
            # File size limit
            if document.file_size and document.file_size > 20 * 1024 * 1024:
                await self.safe_send(chat_id, "❌ File is too large. Maximum size is 20MB.")
                return

            file_name = document.file_name or f"file_{document.file_id}"
            file_obj = await self.bot.get_file(document.file_id)
            local_path = f"data/files/{document.file_id}_{file_name}"
            await file_obj.download_to_drive(local_path)

            # ✅ Check if it's a PDF and auto-caption
            if file_name.lower().endswith(".pdf"):
                caption = message.caption or f"📄 Please read and summarize this PDF: {file_name}"
            else:
                caption = message.caption or f"Please analyze this file: {file_name}"

            # Upload to OpenAI
            uploaded_id = await self.ai.upload_file(local_path)

            # Attach file to the file_search tool
            attachments = [{"file_id": uploaded_id, "tools": [{"type": "file_search"}]}]

            # Send to assistant
            await self.ai.send_user_message_text(thread_id, text=caption, attachments=attachments)
            await self.run_and_reply(chat_id, thread_id)

        except Exception as e:
            logger.error(f"Document handling error: {e}")
            await self.safe_send(chat_id, "❌ Failed to process the document.")
        finally:
            try:
                if local_path and os.path.exists(local_path):
                    os.remove(local_path)
            except Exception as e:
                logger.error(f"Error removing temp document file: {e}")


    async def run_and_reply(self, chat_id: int, thread_id: str):
        try:
            await self.bot.send_chat_action(chat_id, ChatAction.TYPING)
            run = await self.ai.run_and_wait(thread_id, self.assistant_id)
            if run.status != "completed":
                # Provide more detailed error info if available
                error_message = ""
                if run.last_error:
                    error_message = f"*Details*: `{run.last_error.message}`"
                await self.safe_send(chat_id, f"⚠️ Run ended with status: `{run.status}`.{error_message}")
                return
            reply_text = await self.ai.get_last_assistant_message_text(thread_id)
            await self.safe_send(chat_id, reply_text)
        except TimeoutError:
            await self.safe_send(chat_id, "⏳ The assistant took too long. Please try again.")
        except Exception as e:
            logger.error(f"Run/reply error: {e}")
            await self.safe_send(chat_id, "❌ Couldn't get a response. Please try again.")

    async def safe_send(self, chat_id: int, text: str, reply_to: Optional[int] = None):
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_to_message_id=reply_to,
                disable_web_page_preview=True,
            )
        except TelegramError as e:
            # Fallback without markdown if formatting errors occur
            logger.warning(
                f"Markdown send failed: {e}. Sending as plain text.")
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_to_message_id=reply_to,
                    disable_web_page_preview=True
                )
            except TelegramError as final_e:
                logger.error(
                    f"Final send attempt failed for chat {chat_id}: {final_e}")


# =========================
# Entrypoint
# =========================


async def main():
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Optional (Azure/OpenRouter-style)
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    VIP_CODE = os.getenv("VIP_CODE", "VIP123")
    LIMIT = int(os.getenv("LIMIT", 10))
    TIMEOUT = int(os.getenv("TIMEOUT", 30))

    if not BOT_TOKEN or not OPENAI_API_KEY:
        logger.critical(
            "Missing required environment variables: BOT_TOKEN and OPENAI_API_KEY are required.")
        return

    storage = Storage()
    client = AssistantsClient(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    bot = TelegramAssistantsBot(
        bot_token=BOT_TOKEN,
        openai_client=client,
        storage=storage,
        vip_code=VIP_CODE,
        limit=LIMIT,
        timeout=TIMEOUT,
    )

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down.")
    except Exception as e:
        logger.critical(f"Critical error in main loop: {e}", exc_info=True)
    finally:
        await client.close()
        logger.info("Bot has been shut down.")

if __name__ == "__main__":
    asyncio.run(main())
