from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import shutil
import os
import logging
from app.rag.docling_parser import pdf_processor
# IMPORT THE DEPENDENCY, NOT THE INSTANCE
from app.rag.vector_store import get_vector_store, VectorStore

# Initialize router
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ingest", summary="Upload and parse a PDF document")
async def ingest_document(
    file: UploadFile = File(...),
    # Inject VectorStore using Dependency Injection
    v_store: VectorStore = Depends(get_vector_store)
):
    """
    Receives a PDF file, parses it, and indexes it in Qdrant.
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
        result = pdf_processor.parse_pdf(temp_path)
        
        logger.info("Vectorizing and saving to Qdrant...")
        # Use the injected instance 'v_store'
        v_store.add_document(
            filename=result["filename"],
            content=result["content"],
            metadata={"page_count": result["page_count"]}
        )
        
        os.remove(temp_path)
        
        return {
            "status": "success",
            "message": "Document parsed and indexed successfully.",
            "filename": result["filename"],
            "pages": result["page_count"],
            "preview": result["content"][:200] + "..."
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        # If it's our validation error (wrong vector size), pass 400 instead of 500
        if "vector size" in str(e):
             raise HTTPException(status_code=400, detail=str(e))
             
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")