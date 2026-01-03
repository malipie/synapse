from contextlib import asynccontextmanager
from fastapi import FastAPI
from arq import create_pool
from arq.connections import RedisSettings
from app.core.config import settings
# Import shared state to initialize the pool
from app.state import AppState 
# Import routers AFTER state is handled
from app.api.v1 import chat, documents

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    redis_host = settings.REDIS_HOST if hasattr(settings, "REDIS_HOST") else "synapse-redis"
    print(f"🔌 Connecting to Redis for Job Queue at {redis_host}...")
    
    try:
        # Initialize the global pool in the separate state module
        AppState.arq_pool = await create_pool(RedisSettings(host=redis_host, port=6379))
        print("✅ Redis Pool initialized.")
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
    
    yield
    
    # --- Shutdown ---
    if AppState.arq_pool:
        await AppState.arq_pool.close()
        print("🔌 Redis Pool closed.")

app = FastAPI(
    title="Synapse API",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(chat.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")