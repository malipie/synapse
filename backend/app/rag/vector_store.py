import uuid
import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
from pydantic import ValidationError
from fastembed import TextEmbedding, SparseTextEmbedding
import openai
import numpy as np 

from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.collection_name = "documents"
        self.provider = settings.EMBEDDING_PROVIDER
        
        # 1. DENSE VECTOR CONFIG
        if self.provider == "openai":
            self.dense_vector_size = settings.VECTOR_SIZE_OPENAI
        else:
            self.dense_vector_size = settings.VECTOR_SIZE_LOCAL

        logger.info(f"Initializing VectorStore with provider: {self.provider} (Dense Dim: {self.dense_vector_size})")

        # 2. QDRANT CLIENT
        try:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST, 
                port=settings.QDRANT_PORT,
                timeout=60.0
            )
        except Exception as e:
            logger.critical(f"Failed to connect to Qdrant: {e}")
            raise e
        
        # 3. INITIALIZE MODELS
        self.embedding_model = None
        self.openai_client = None
        self.sparse_embedding_model = None 
        
        try:
            if self.provider == "openai":
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY is missing.")
                self.openai_client = openai.OpenAI(
                    api_key=settings.OPENAI_API_KEY, timeout=30.0, max_retries=3
                )
            else:
                logger.info("Loading Dense model 'BAAI/bge-small-en-v1.5'...")
                self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                logger.info("Dense model loaded.")
            
            logger.info("Loading Sparse model 'prithivida/Splade_pp_en_v1'...")
            self.sparse_embedding_model = SparseTextEmbedding(model_name="prithivida/Splade_pp_en_v1", threads=1)
            logger.info("Sparse model loaded.")

            self._validate_or_create_collection()
            
        except Exception as e:
            logger.critical(f"Failed to initialize VectorStore components: {e}")
            raise e

    def _validate_or_create_collection(self):
        """
        Self-Healing Collection Creator.
        Robustly handles version mismatches between client and server.
        """
        try:
            # 1. Sprawdź, czy kolekcja w ogóle istnieje (to zazwyczaj nie rzuca błędów walidacji)
            collections_response = self.client.get_collections()
            collection_names = [c.name for c in collections_response.collections]

            should_recreate = False

            if self.collection_name in collection_names:
                # 2. Jeśli istnieje, spróbuj pobrać jej konfigurację
                try:
                    info = self.client.get_collection(self.collection_name)
                    
                    # Sprawdź czy posiada konfigurację dla 'text-sparse'
                    sparse_config = info.config.params.sparse_vectors
                    
                    if not sparse_config or "text-sparse" not in sparse_config:
                        logger.warning(f"Collection schema outdated (no sparse vectors). Scheduling recreation...")
                        should_recreate = True
                    else:
                        logger.info(f"Collection validated successfully.")

                except (ValidationError, ResponseHandlingException) as ve:
                    # 3. CRITICAL FIX: Złap błąd walidacji Pydantic
                    logger.warning(f"Version mismatch detected (Client vs Server schema): {ve}. Forcing recreation to fix state.")
                    should_recreate = True
            else:
                should_recreate = True

            # 4. Wykonaj reset jeśli trzeba
            if should_recreate:
                logger.info(f"Recreating Hybrid Collection '{self.collection_name}'...")
                
                # Usuń starą (ignoruj błędy jeśli nie istnieje)
                try:
                    self.client.delete_collection(self.collection_name)
                except Exception:
                    pass

                # Stwórz nową
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.dense_vector_size,
                        distance=models.Distance.COSINE
                    ),
                    sparse_vectors_config={
                        "text-sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(
                                on_disk=True,
                            )
                        )
                    }
                )
                logger.info(f"Hybrid Collection created successfully.")
                
        except Exception as e:
            logger.error(f"Error during collection validation/creation: {e}")
            # Nie rzucamy 'raise', żeby aplikacja mogła wstać, nawet jeśli baza ma czkawkę
            
    def _get_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            response = self.openai_client.embeddings.create(
                input=texts, model="text-embedding-3-small"
            )
            return [data.embedding for data in response.data]
        else:
            embeddings = list(self.embedding_model.embed(texts))
            return [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]

    def _get_sparse_embeddings(self, texts: List[str]) -> List[Any]:
        return list(self.sparse_embedding_model.embed(texts))

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        if not text: return []
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                last_newline = text.rfind('\n', start, end)
                if last_newline != -1 and last_newline > start + (chunk_size // 2):
                    end = last_newline + 1
                else:
                    last_space = text.rfind(' ', start, end)
                    if last_space != -1 and last_space > start + (chunk_size // 2):
                        end = last_space + 1
            chunk = text[start:end].strip()
            if len(chunk) >= 5: chunks.append(chunk)
            start += chunk_size - overlap
            if start >= text_len: break
        return chunks

    def add_document(self, filename: str, content: str, metadata: Dict[str, Any] = None):
        if metadata is None: metadata = {}
        logger.info(f"Processing document: {filename} (Hybrid Mode)")
        
        try:
            all_chunks = self._chunk_text(content)
            if not all_chunks: return

            total_chunks = len(all_chunks)
            BATCH_SIZE = 10 
            
            logger.info(f"Total chunks: {total_chunks}. Processing in batches of {BATCH_SIZE}...")

            for i in range(0, total_chunks, BATCH_SIZE):
                batch_chunks = all_chunks[i : i + BATCH_SIZE]
                
                dense_embeddings = self._get_dense_embeddings(batch_chunks)
                sparse_embeddings = self._get_sparse_embeddings(batch_chunks)

                points = []
                for j, (chunk, dense, sparse) in enumerate(zip(batch_chunks, dense_embeddings, sparse_embeddings)):
                    absolute_index = i + j
                    
                    if isinstance(dense, np.ndarray):
                        dense = dense.tolist()
                        
                    sparse_vector = models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist()
                    )

                    payload = {
                        "filename": filename,
                        "content": chunk,
                        "chunk_index": absolute_index,
                        "total_chunks": total_chunks,
                        **metadata
                    }
                    
                    points.append(models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "": dense, 
                            "text-sparse": sparse_vector
                        },
                        payload=payload
                    ))

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Indexed batch {i//BATCH_SIZE + 1}/{(total_chunks + BATCH_SIZE - 1)//BATCH_SIZE}")

            logger.info(f"Successfully indexed all {total_chunks} chunks for {filename}.")
            
        except Exception as e:
            logger.error(f"Failed to add document {filename}: {e}")
            raise e

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            dense_query = self._get_dense_embeddings([query])[0]
            if isinstance(dense_query, np.ndarray):
                dense_query = dense_query.tolist()

            sparse_query_raw = self._get_sparse_embeddings([query])[0]
            
            sparse_query = models.SparseVector(
                indices=sparse_query_raw.indices.tolist(),
                values=sparse_query_raw.values.tolist()
            )

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_query,
                        using=None,
                        limit=limit * 2
                    ),
                    models.Prefetch(
                        query=sparse_query,
                        using="text-sparse",
                        limit=limit * 2
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit
            )

            results = []
            for hit in search_result.points:
                results.append({
                    "content": hit.payload.get("content"),
                    "filename": hit.payload.get("filename"),
                    "score": hit.score,
                    "metadata": hit.payload
                })
            
            return results

        except Exception as e:
            logger.error(f"Hybrid Search failed: {e}")
            return []

@lru_cache()
def get_vector_store() -> VectorStore:
    return VectorStore()