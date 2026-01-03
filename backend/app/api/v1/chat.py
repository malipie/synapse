import logging
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Security
from app.core.llm_service import secure_llm
# CRITICAL FIX: Import AppState from 'state.py', NOT 'main.py' to avoid circular import
from app.state import AppState

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[Dict[str, str]]] = []

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None

# --- Endpoints ---

@router.post("/chat/enqueue", response_model=TaskResponse, status_code=202)
async def enqueue_chat_task(request: ChatRequest):
    if not AppState.arq_pool:
        raise HTTPException(status_code=503, detail="Task queue not available")

    try:
        raw_query = request.message
        safe_query = secure_llm._sanitize_input(raw_query)
        
        # Enqueue job using the pool from AppState
        job = await AppState.arq_pool.enqueue_job('run_agent_workflow', safe_query)
        
        return TaskResponse(
            task_id=job.job_id,
            status="queued",
            message="Agent workflow started in background."
        )

    except Exception as e:
        logger.error(f"Enqueue failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    if not AppState.arq_pool:
        raise HTTPException(status_code=503, detail="Task queue not available")

    try:
        from arq.jobs import Job, JobStatus
        
        job = Job(task_id, AppState.arq_pool)
        status = await job.status()

        # Normalize status to a simple string like 'queued', 'started', 'complete', 'failed'
        try:
            status_str = status.value if hasattr(status, 'value') else str(status)
        except Exception:
            status_str = str(status)

        # In case str(status) returns 'JobStatus.complete', take last token
        if isinstance(status_str, str) and '.' in status_str:
            status_str = status_str.split('.')[-1]

        status_str = status_str.lower()

        response = TaskStatusResponse(task_id=task_id, status=status_str)

        if status_str == 'complete':
            response.result = await job.result()
        elif status_str == 'failed':
            response.error = "Task execution failed."
            
        return response

    except Exception as e:
        logger.warning(f"Task check error: {e}")
        return TaskStatusResponse(task_id=task_id, status="not_found", error="Task not found")