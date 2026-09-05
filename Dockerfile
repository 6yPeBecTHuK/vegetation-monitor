FROM python:3.11-slim

WORKDIR /app

# Системные зависимости (если нужны для numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект (src, webapp, data, artifacts если есть)
COPY src ./src
COPY webapp ./webapp
COPY data ./data

# Копируем артефакты обучения, если они закоммичены
COPY artifacts ./artifacts

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Запуск
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "$PORT"]