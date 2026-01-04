from typing import Annotated, Callable, Any
from app.rag.vector_store import VectorStore

def get_search_tool(vector_store: VectorStore) -> Callable:
    """
    Tworzy funkcję narzędziową dla AutoGena z wstrzykniętą bazą wektorową.
    """
    
    def search_documents(
        query: Annotated[str, "Keywords to search. Do NOT use full sentences."]
    ) -> str:
        # Logowanie wewnątrz narzędzia pomaga w debugowaniu
        print(f"🔎 [Tool] Researcher searching for: '{query}'")
        try:
            results = vector_store.search(query)
            
            if not results:
                return "No documents found containing these keywords."
            
            formatted_results = []
            for res in results:
                # Obsługa różnych formatów zwracanych przez Qdrant
                content = None
                
                # Próba wyciągnięcia tekstu z różnych miejsc
                if isinstance(res, dict):
                    content = res.get('text') or res.get('content') or res.get('page_content')
                    if not content and 'payload' in res:
                        payload = res['payload']
                        if isinstance(payload, dict):
                            content = payload.get('text') or payload.get('content')
                elif hasattr(res, 'payload'):
                    payload = res.payload
                    if isinstance(payload, dict):
                        content = payload.get('text') or payload.get('content')
                
                if not content: 
                    content = str(res) # Fallback

                clean_content = " ".join(content.split())[:2000]
                
                filename = "Unknown Source"
                if isinstance(res, dict):
                    meta = res.get('metadata') or res.get('payload') or {}
                    filename = meta.get('filename') or meta.get('source') or "Unknown"
                elif hasattr(res, 'payload') and isinstance(res.payload, dict):
                    filename = res.payload.get('filename') or "Unknown"
                
                formatted_results.append(f"Source: {filename}\nContent: {clean_content}")

            return "\n\n".join(formatted_results)

        except Exception as e:
            return f"Search Error: {str(e)}"

    return search_documents