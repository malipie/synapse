import os
import sys

# Dodajemy katalog nadrzędny do ścieżki, żeby widzieć moduły aplikacji (opcjonalne, tu używamy tylko SDK)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langfuse import Langfuse
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych (jeśli uruchamiasz lokalnie z terminala)
# Upewnij się, że masz ustawione LANGFUSE_PUBLIC_KEY, SECRET_KEY i HOST
load_dotenv()

def init_prompts():
    print("🚀 Initializing Langfuse Prompts...")
    
    # Inicjalizacja klienta
    langfuse = Langfuse()
    
    # 1. Router Intent
    print(" -> Creating 'synapse-router'...")
    langfuse.create_prompt(
        name="synapse-router",
        prompt="Classify user intent: 'RAG' or 'CHAT'. Return ONE word.",
        config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "supported_variables": [] 
        },
        labels=["production", "core"]
    )

    # 2. Small Talk (Chat)
    print(" -> Creating 'synapse-smalltalk'...")
    langfuse.create_prompt(
        name="synapse-smalltalk",
        prompt="Jesteś asystentem Synapse. Odpowiadaj krótko i rzeczowo.",
        config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "supported_variables": []
        },
        labels=["production", "chat"]
    )

    # 3. Researcher (AutoGen)
    print(" -> Creating 'synapse-researcher'...")
    langfuse.create_prompt(
        name="synapse-researcher",
        prompt=(
            "You are a Senior Medical Regulatory Analyst. "
            "Your goal is to answer the user's question based ONLY on the search results provided. "
            "1. Search for keywords.\n"
            "2. ANALYZE the results provided by the Admin.\n"
            "3. Write a summary answering the user's question.\n"
            "4. If the user asks in Polish, ANSWER IN POLISH.\n"
        ),
        config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
             "supported_variables": []
        },
        labels=["production", "agent"]
    )

    # 4. Critic (AutoGen)
    print(" -> Creating 'synapse-critic'...")
    langfuse.create_prompt(
        name="synapse-critic",
        prompt=(
            "You are a Quality Assurance Auditor. "
            "Check the Researcher's answer. "
            "If it answers the question and is in the correct language, say 'TERMINATE'. "
            "Otherwise, tell them what to fix."
        ),
        config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "supported_variables": []
        },
        labels=["production", "agent"]
    )

    print("✅ All prompts created successfully!")

if __name__ == "__main__":
    try:
        init_prompts()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Hint: Ensure LANGFUSE_HOST, PUBLIC_KEY, and SECRET_KEY are set.")