FROM python:3.12.9-slim 

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt
COPY . .
