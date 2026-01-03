import logging
from typing import List, Annotated
from app.rag.vector_store import VectorStore

# Configure logging
logger = logging.getLogger(__name__)

# global instance or a way to inject the store. 


def search_documents_tool(
    query: Annotated[str, "The search query to look up in the medical documents"],
    vector_store: VectorStore
) -> str:
    """
    Searches the vector database for relevant document chunks based on the query.
    Returns the content found to be used as context.
    """
    try:
        logger.info(f"Agent Tool invoked: Searching for '{query}'")
        
        # Perform retrieval (Hybrid Search provided by the backend core)
        results = vector_store.search(query, limit=5)
        
        if not results:
            return "No relevant information found in the documents."
            
        # Format the results for the Agent
        context_str = ""
        for i, res in enumerate(results, 1):
            source = res['metadata'].get('filename', 'Unknown Source')
            content = res['content']
            context_str += f"Result {i} (Source: {source}):\n{content}\n\n"
            
        return context_str

    except Exception as e:
        logger.error(f"Search tool error: {str(e)}")
        return f"Error occurred during search: {str(e)}"