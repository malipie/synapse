from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.core.llm_service import get_secure_llm 
from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job, JobStatus 
from app.core.config import settings
import os

router = APIRouter()

class ChatRequest(BaseModel):
    messages: List[dict]
    model: Optional[str] = "gpt-3.5-turbo"

@router.post("/")
async def chat_endpoint(request: ChatRequest):
    try:
        user_query = request.messages[-1]["content"]
        secure_llm = get_secure_llm()

        # 1. Classyfication
        intent = await secure_llm.classify_intent(user_query)
        
        # 2. CHAT
        if intent == "CHAT":
            response = await secure_llm.get_chat_response(request.messages)
            return {"role": "assistant", "content": response, "intent": "CHAT"}
        
        # 3. RAG
        redis_host = getattr(settings, "REDIS_HOST", "synapse-redis")
        redis = await create_pool(RedisSettings(host=redis_host, port=6379))
        
        job = await redis.enqueue_job("run_agent_workflow", user_query)
        await redis.close()
        
        return {
            "role": "assistant", 
            "content": "Rozpoczynam analizę dokumentów...",
            "job_id": job.job_id,
            "intent": "RAG"
        }

    except Exception as e:
        print(f"❌ CHAT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{job_id}")
async def get_task_status(job_id: str):
    try:
        redis_host = getattr(settings, "REDIS_HOST", "synapse-redis")
        redis = await create_pool(RedisSettings(host=redis_host, port=6379))
        
        job = Job(job_id, redis)
        status = await job.status()
        
        result = None
        if status == JobStatus.complete:
            result = await job.result()
            
        await redis.close()
        
        return {
            "job_id": job_id,
            "status": status,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
@router.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "mock_mode": os.getenv("MOCK_LLM", "false")
    }