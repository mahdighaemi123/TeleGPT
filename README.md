# Telegram AI Assistant Bot

A Telegram bot powered by OpenAI for AI-based conversations. Features VIP user access, persistent conversation memory, and easy configuration.

---

## Features

- **AI-Powered Chat:** OpenAI-powered responses in Telegram.
- **VIP User System:** Only whitelisted users (VIP code) can chat with the bot.
- **Persistent Memory:** Maintains chat histories for context-aware conversations.
- **Simple Configuration:** All settings via `.env` file.
- **Docker Support:** Deploy with Docker Compose.

---

## Quick Start

### 1. Clone the Repo

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 2. Create Environment Variables

Copy the template and fill in your credentials:

```bash
cp .env.template .env
# Edit .env with your bot token, OpenAI details, and VIP code
```

**Sample `.env` variables:**

```dotenv
BOT_TOKEN=replace_with_your_telegram_token
OPENAI_API_KEY=replace_with_your_openai_key
OPENAI_API_BASE=replace_with_your_openai_base_url
VIP_CODE=replace_with_a_secret_code
MODEL_NAME=gpt-4.1-mini
MAX_TOKENS=1000
TEMPERATURE=0.7
LIMIT=10
TIMEOUT=30
```

---

## Running with Docker Compose

> **Recommended:** Ensure you have Docker and Docker Compose installed.

Build and run the bot with:

```bash
docker compose up --build -d
```

- Stops/starts the bot automatically (`restart: always`)
- Persists data (VIP list, memory, etc.) in the `./data` directory
- Uses a custom Docker bridge network

To view logs:

```bash
docker compose logs -f
```

To stop the bot:

```bash
docker compose down
```

---

## Directory Structure

```
.
├── data/           # Persistent data: VIPs, memory, etc.
├── bot.py          # Main script
├── Dockerfile/     # (If you have one; required for Docker builds)
├── docker-compose.yml
├── requirements.txt
├── .env
├── .env.template
└── README.md
```

---

## Usage

1. **Chat with your bot** on Telegram.
2. **Send the VIP code** you set in `.env` to get access as a VIP.
3. **Now you can use the bot:** Send messages, get AI replies with context and memory!

---

## Contributing

Pull requests and issues are welcome. Feel free to improve or fix any part of the bot.

---