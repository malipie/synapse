import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Synapse"
    
    # OPENAI Configuration
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    # Qdrant Configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "synapse-qdrant")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    
    # Embedding Configuration
    # Options: "local" (FastEmbed) or "openai"
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local") 
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Vector Dimensions
    # BAAI/bge-small-en-v1.5 outputs 384 dimensions
    VECTOR_SIZE_LOCAL: int = 384 
    # OpenAI text-embedding-3-small outputs 1536 dimensions
    VECTOR_SIZE_OPENAI: int = 1536

settings = Settings()