import asyncio
import logging
from typing import Any, Dict

from arq import Worker
from arq.connections import RedisSettings

from app.core.config import settings
from app.agents.medical_agent import MedicalAgentTeam
from app.rag.vector_store import get_vector_store

# Configure logging
logger = logging.getLogger(__name__)

async def startup(ctx: Dict[str, Any]) -> None:
    """
    Initialize resources when the worker starts.
    We initialize the VectorStore here to avoid recreating it for every job.
    """
    logger.info("🚀 Worker starting up... Initializing dependencies.")
    # Initialize VectorStore singleton-like for the worker process
    # Note: In a real prod env, we might want dependency injection, 
    # but for Arq worker context, explicit initialization is common.
    ctx['vector_store'] = get_vector_store()
    logger.info("✅ Worker dependencies initialized.")

async def shutdown(ctx: Dict[str, Any]) -> None:
    """Cleanup resources on shutdown."""
    logger.info("🛑 Worker shutting down...")

async def run_agent_workflow(ctx: Dict[str, Any], query: str) -> str:
    """
    The main background task.
    Executes the MedicalAgentTeam workflow.
    """
    logger.info(f"👷 Worker executing task for query: {query}")
    
    try:
        # Retrieve initialized vector store from context
        v_store = ctx['vector_store']
        
        # Initialize the Agent Team
        # NOTE: Ideally MedicalAgentTeam should be refactored to be fully async in the future.
        # For now, if run() is synchronous, it blocks this specific worker coroutine.
        agent_team = MedicalAgentTeam(vector_store=v_store)
        
        # Execute workflow (blocking call wrapped if necessary, or native if refactored)
        # Since MedicalAgentTeam.run is currently sync, we run it directly here.
        # In high-load asyncio, we would use run_in_executor to avoid blocking the event loop.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, agent_team.run, query)
        
        logger.info("✅ Task completed successfully.")
        return result

    except Exception as e:
        logger.error(f"❌ Task failed: {e}")
        raise e

# Arq Worker Configuration
class WorkerSettings:
    # Connect to the Redis container named 'synapse-redis'
    # Fallback to localhost for local dev without docker networking
    redis_settings = RedisSettings(
        host=settings.REDIS_HOST if hasattr(settings, "REDIS_HOST") else "synapse-redis",
        port=6379
    )
    
    # Register functions
    functions = [run_agent_workflow]
    
    # Lifecycle hooks
    on_startup = startup
    on_shutdown = shutdown
    
    # Queue settings
    max_jobs = 10