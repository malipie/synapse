import asyncio
import os
import logging
from typing import Any, Dict

from arq import Worker
from arq.connections import RedisSettings

# --- CONFIG LITELLM ---
import litellm
# Debug logs (opcjonalnie, można wyłączyć w produkcji)
os.environ["LITELLM_LOG"] = "INFO"

# Podpięcie Langfuse
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
# ----------------------

from app.core.config import settings
from app.agents.medical_agent import MedicalAgentTeam
from app.rag.vector_store import get_vector_store
from app.core.llm_service import secure_llm

logger = logging.getLogger(__name__)

async def startup(ctx: Dict[str, Any]) -> None:
    logger.info("🚀 Worker starting up...")
    
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        logger.error("❌ LANGFUSE_PUBLIC_KEY is missing!")
    
    ctx['vector_store'] = get_vector_store()
    logger.info("✅ Worker ready.")

async def shutdown(ctx: Dict[str, Any]) -> None:
    logger.info("🛑 Worker shutting down...")

async def run_agent_workflow(ctx: Dict[str, Any], query: str) -> str:
    logger.info(f"👷 Processing task: {query}")
    try:
        # 1. ROUTER
        intent = await secure_llm.classify_intent(query)
        logger.info(f"🧠 Intent: {intent}")

        response = ""
        
        # Ścieżka A: CHAT
        if intent == "CHAT":
            messages = [
                {"role": "system", "content": "Jesteś asystentem Synapse. Odpowiadaj krótko."},
                {"role": "user", "content": query}
            ]
            response = await secure_llm.get_chat_response(messages)
            
        # Ścieżka B: RAG
        else:
            logger.info("📚 Running RAG...")
            v_store = ctx['vector_store']
            agent_team = MedicalAgentTeam(vector_store=v_store)
            
            loop = asyncio.get_running_loop()
            # Uruchomienie agentów
            response = await loop.run_in_executor(None, agent_team.run, query)

        # --- FIX: Bezpieczny Flush ---
        # Usuwamy litellm.utils.flush_cache()
        # Dajemy 1s na wysłanie logów w tle przez SDK Langfuse
        await asyncio.sleep(1)
        # -----------------------------
        
        return response

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return f"System Error: {str(e)}"

class WorkerSettings:
    redis_settings = RedisSettings(
        host=settings.REDIS_HOST if hasattr(settings, "REDIS_HOST") else "synapse-redis",
        port=6379
    )
    functions = [run_agent_workflow]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 10