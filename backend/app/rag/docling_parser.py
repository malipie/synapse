import logging
from pathlib import Path
from typing import Dict, Any, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Configure logging
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles the ingestion and parsing of PDF documents using Docling.
    Implements Lazy Loading to prevent blocking server startup.
    """

    def __init__(self):
        # Lazy loading pattern: Do not instantiate the heavy converter immediately.
        # This allows the API to start up quickly without waiting for PyTorch models to load.
        self._converter: Optional[DocumentConverter] = None

    def _get_converter(self) -> DocumentConverter:
        """
        Singleton-like accessor for the DocumentConverter.
        Initializes the model only on the first request.
        """
        if self._converter is None:
            logger.info("Initializing Docling models (First run)... This may take a few seconds.")
            try:
                # --- FAST MODE CONFIGURATION START ---
                pipeline_options = PdfPipelineOptions()
                
                # 1. Disable OCR (Use text layer directly)
                # Drastically reduces CPU load.
                pipeline_options.do_ocr = False 
                
                # 2. Disable Table Structure AI (TableFormer)
                # This skips the heavy neural network analysis for tables.
                pipeline_options.do_table_structure = False
                
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
                # --- FAST MODE CONFIGURATION END ---
                
                logger.info("Docling models loaded successfully (FAST MODE enabled).")
            except Exception as e:
                logger.critical(f"Failed to initialize Docling: {e}")
                raise RuntimeError("Could not initialize document parser.")
        return self._converter

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a PDF file and returns a structured representation.
        
        Args:
            file_path: Absolute path to the PDF file.
            
        Returns:
            Dict containing metadata and the full text/structure.
        """
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Document at {file_path} does not exist.")

        logger.info(f"Starting Docling extraction for: {path_obj.name}")
        
        try:
            # Get the converter (loads model if not already loaded)
            converter = self._get_converter()
            
            # The heavy lifting happens here
            result = converter.convert(path_obj)
            document = result.document
            
            # Export to Markdown is usually the best format for LLMs (preserves headers/tables)
            markdown_content = document.export_to_markdown()
            
            # Basic stats
            logger.info(f"Successfully parsed {path_obj.name}. Pages: {len(document.pages)}")
            
            return {
                "filename": path_obj.name,
                "page_count": len(document.pages),
                "content": markdown_content,  # This is what goes into the LLM/Vector DB
                "metadata": {
                    "source": str(path_obj),
                    "parser": "docling_fast"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {str(e)}")
            raise RuntimeError(f"Docling parsing failed: {str(e)}")

# Singleton instance for easy import
pdf_processor = PDFProcessor()