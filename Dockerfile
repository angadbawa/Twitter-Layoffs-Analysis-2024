FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
COPY src/ ./src/
COPY config/ ./config/
COPY notebooks/ ./notebooks/
RUN mkdir -p /app/output
COPY *.csv ./data/ 2>/dev/null || true
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app
EXPOSE 8888
CMD ["python", "-c", "from src.twitter_analysis import *; print('Twitter Analysis Environment Ready!')"]

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src.twitter_analysis; print('OK')" || exit 1
