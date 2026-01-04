FROM python:3.11-slim

WORKDIR /app

# Ustawienia środowiskowe
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- WARSTWA 1: Zależności systemowe (Rzadko zmieniane) ---
# Instalujemy kompilatory i curla
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- WARSTWA 2: Narzędzia Python (Rzadko zmieniane) ---
RUN pip install --upgrade pip uv

# --- WARSTWA 3: Zależności Python (Zmieniane okazjonalnie) ---
# Kopiujemy TYLKO plik requirements.txt
COPY requirements.txt .

# Instalujemy biblioteki. Jeśli requirements.txt się nie zmieni, Docker weźmie to z cache.
RUN uv pip install --system --no-cache -r requirements.txt "numpy<2.0.0"

# --- WARSTWA 4: Modele AI (Bardzo rzadko zmieniane) ---
RUN python -m spacy download pl_core_news_md && \
    python -m spacy download en_core_web_lg

# --- WARSTWA 5: Kod Aplikacji (Zmieniany często) ---
# Kopiujemy resztę kodu na samym końcu.
# Zmiana w worker.py wpłynie tylko na tę warstwę i kolejne (CMD).
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]