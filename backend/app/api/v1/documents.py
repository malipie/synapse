from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import shutil
import os
import logging
from app.rag.docling_parser import pdf_processor
# [FIX] Importujemy get_secure_llm
from app.core.llm_service import get_secure_llm 
from app.rag.vector_store import get_vector_store, VectorStore

# Initialize router
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ingest", summary="Upload and parse a PDF document")
async def ingest_document(
    file: UploadFile = File(...),
    v_store: VectorStore = Depends(get_vector_store)
):
    """
    Receives a PDF file, parses it via Docling, MASKS PII, and indexes it in Qdrant.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_dir = "/tmp/ingest"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        logger.info(f"Receiving file: {file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info("Starting Docling parsing...")
        # 1. Parsowanie PDF (Zostawiamy Twoją logikę)
        result = pdf_processor.parse_pdf(temp_path)
        
        # 2. [NOWOŚĆ] Maskowanie PII w wyciągniętym tekście
        logger.info("Sanitizing content (PII Masking)...")
        secure_llm = get_secure_llm()
        
        # Wyciągamy tekst, maskujemy go i podmieniamy w obiekcie result
        original_content = result["content"]
        # Używamy _sanitize_input (zwraca string)
        safe_content = secure_llm._sanitize_input(original_content)
        
        logger.info("Vectorizing and saving to Qdrant...")
        v_store.add_document(
            filename=result["filename"],
            content=safe_content, # Zapisujemy bezpieczną wersję
            metadata={"page_count": result["page_count"], "pii_masked": True}
        )
        
        os.remove(temp_path)
        
        return {
            "status": "success",
            "message": "Document parsed, sanitized and indexed successfully.",
            "filename": result["filename"],
            "pages": result["page_count"],
            "preview": safe_content[:200] + "..."
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        if "vector size" in str(e):
             raise HTTPException(status_code=400, detail=str(e))
             
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")