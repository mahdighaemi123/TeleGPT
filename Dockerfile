FROM python:3.11-alpine

# Install build essentials, gcc, and common dependencies
RUN apk update && \
    apk add --no-cache \
    build-base \
    gcc \
    libc-dev \
    libffi-dev \
    openssl-dev \
    bash \
    curl \
    git

WORKDIR /app

# install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the default command to run your Python script
CMD ["python", "bot.py"]
