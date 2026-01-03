FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

COPY requirements.txt .

# Instalacja z wymuszeniem wersji
RUN uv pip install --system --no-cache -r requirements.txt "numpy<2.0.0"



# Download Polish Spacy model
RUN python -m spacy download pl_core_news_md && \
    python -m spacy download en_core_web_lg

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]