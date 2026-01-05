FROM python:3.11-slim

WORKDIR /app

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- LAYER 1: System dependencies (Rarely changed) ---
# Install compilers and curl
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- LAYER 2: Python tools (Rarely changed) ---
RUN pip install --upgrade pip uv

# --- LAYER 3: Python dependencies (Changed occasionally) ---
# Copy ONLY requirements.txt file
COPY requirements.txt .

# Install libraries. If requirements.txt hasn't changed, Docker will use the cache.
RUN uv pip install --system --no-cache -r requirements.txt "numpy<2.0.0"

# --- LAYER 4: AI Models (Very rarely changed) ---
RUN python -m spacy download pl_core_news_md && \
    python -m spacy download en_core_web_lg

# --- LAYER 5: Application Code (Changed frequently) ---
# Copy the rest of the code at the very end.
# Changes in worker.py will only affect this layer and subsequent ones (CMD).
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]