FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

# ZMIANA: Dodano --system. To naprawia błąd "No virtual environment found".
RUN pip install uv && uv pip install --system --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]