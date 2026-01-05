import asyncio
import os
import logging
from typing import Any, Dict

from arq.connections import RedisSettings
import litellm

# --- Config ---
os.environ["LITELLM_LOG"] = "INFO"
# Langfuse callbacks (optional, safe mode)
if os.getenv("LANGFUSE_PUBLIC_KEY"):
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

# Import agent logic
from app.agents.medical_agent import MedicalAgentTeam
from app.rag.vector_store import get_vector_store

# [FIX] Safe settings import - if it fails, worker starts anyway
try:
    from app.core.config import settings
except ImportError:
    settings = None
except Exception as e:
    print(f"⚠️ Warning: Settings loading failed ({e}), falling back to env vars.")
    settings = None

# [FIX] Import get_secure_llm function
from app.core.llm_service import get_secure_llm 

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# [FIX] Hardcoded Redis config from env (independent of Pydantic)
REDIS_HOST = os.getenv("REDIS_HOST", "synapse-redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

async def startup(ctx: Dict[str, Any]) -> None:
    logger.info(f"🚀 [Worker] Starting up... Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    
    # Check Langfuse
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        logger.warning("⚠️ LANGFUSE_PUBLIC_KEY is missing! Observability might not work.")
    
    # Initialize Vector Store
    try:
        ctx['vector_store'] = get_vector_store()
        logger.info("✅ [Worker] VectorStore initialized.")
    except Exception as e:
        logger.error(f"❌ [Worker] VectorStore init failed: {e}")

    logger.info("✅ [Worker] Ready to process jobs.")

async def shutdown(ctx: Dict[str, Any]) -> None:
    logger.info("🛑 [Worker] Shutting down...")

async def run_agent_workflow(ctx: Dict[str, Any], query: str) -> str:

    if os.getenv("MOCK_LLM") == "true":
        logger.info(f"🎭 [MOCK MODE] Simulating RAG for: {query}")
        await asyncio.sleep(2) 
        return f"To jest symulowana odpowiedź RAG dla zapytania: {query}. Dokumenty zostały (teoretycznie) przeszukane."
    
    logger.info(f"👷 [Job Start] Processing task: {query}")
    
    try:
        # [FIX] Get LLM instance (Lazy Loading)
        secure_llm = get_secure_llm()
        
        # 1. Classification
        intent = await secure_llm.classify_intent(query)
        logger.info(f"🧠 Intent detected: {intent}")

        response = ""
        if intent == "CHAT":
            # Small Talk
            messages = [{"role": "user", "content": query}]
            response = await secure_llm.get_chat_response(messages)
            
        else:
            # RAG Workflow
            logger.info("📚 Running RAG Pipeline...")
            v_store = ctx['vector_store']
            agent_team = MedicalAgentTeam(vector_store=v_store)
            
            # Run agents in executor (since AutoGen can be blocking)
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, agent_team.run, query)

        logger.info("✅ [Job Complete] Response generated.")
        return response

    except Exception as e:
        logger.error(f"❌ [Job Error]: {e}", exc_info=True)
        return f"System Error: {str(e)}"

# --- ARQ Settings ---
class WorkerSettings:
    # [FIX] Use variables defined above, not settings.*
    redis_settings = RedisSettings(host=REDIS_HOST, port=REDIS_PORT)
    
    functions = [run_agent_workflow]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 10
    # Important for Docker stability
    handle_signals = False