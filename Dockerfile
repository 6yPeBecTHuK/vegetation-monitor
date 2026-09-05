FROM python:3.11-slim

WORKDIR /app

# libgomp1 нужен LightGBM на Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY webapp ./webapp
COPY data ./data

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY artifacts ./artifacts
# ВАЖНО: shell-форма (БЕЗ квадратных скобок) —
# только так sh раскроет $PORT в число
CMD uvicorn webapp.app:app --host 0.0.0.0 --port $PORT