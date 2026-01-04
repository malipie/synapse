from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from arq import create_pool
from arq.connections import RedisSettings
import logging

from app.core.config import settings
from app.state import AppState 
# Importujemy poprawione routery
from app.api.v1 import chat, documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    redis_host = settings.REDIS_HOST if hasattr(settings, "REDIS_HOST") else "synapse-redis"
    logger.info(f"🔌 Connecting to Redis at {redis_host}...")
    
    try:
        AppState.arq_pool = await create_pool(RedisSettings(host=redis_host, port=6379))
        logger.info("✅ Redis Pool initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
    
    yield
    
    # --- Shutdown ---
    if AppState.arq_pool:
        await AppState.arq_pool.close()
        logger.info("🔌 Redis Pool closed.")

app = FastAPI(
    title="Synapse API",
    version="1.0.0",
    lifespan=lifespan,
    openapi_url="/api/v1/openapi.json" # Ważne dla Swaggera
)

# CORS - Pozwalamy na wszystko w trybie dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REJESTRACJA ROUTERÓW (POPRAWIONA) ---
# Teraz backend będzie słuchał tam, gdzie Streamlit puka

# 1. Chat: /api/v1/chat
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

# 2. Documents: /api/v1/documents
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])