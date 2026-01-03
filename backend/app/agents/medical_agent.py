import logging
import os
from typing import Annotated

import autogen
# Import LiteLLM do obsługi Langfuse wewnątrz AutoGen
from litellm import completion 
from app.rag.vector_store import VectorStore
from app.core.config import settings

logger = logging.getLogger(__name__)

class MedicalAgentTeam:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        # Konfiguracja LLM
        config_list = [{
            "model": settings.OPENAI_MODEL_NAME if hasattr(settings, "OPENAI_MODEL_NAME") else "gpt-3.5-turbo",
            "api_key": settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"),
            "tags": ["autogen", "medical-agent"],
        }]

        self.llm_config = {
            "config_list": config_list,
            "temperature": 0.1, 
        }

    def run(self, user_query: str) -> str:
        logger.info(f"🚀 Starting Agent Run for: {user_query}")
        try:
            # 1. Observability
            import litellm
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["langfuse"]

            # 2. Narzędzie RAG
            def search_documents(query: Annotated[str, "Keywords"]) -> str:
                print(f"🔎 DEBUG: Searching: '{query}'")
                try:
                    results = self.vector_store.search(query)
                    if not results: return "No documents found."
                    
                    # Formatowanie
                    formatted = []
                    for res in results:
                        content = str(res.get('content') or res.get('text') or str(res))
                        clean = " ".join(content.split())[:2000]
                        formatted.append(f"Content: {clean}")
                    return "\n\n".join(formatted)
                except Exception as e:
                    return f"Error: {e}"

            # 3. Agenci
            user_proxy = autogen.UserProxyAgent(
                name="Admin",
                system_message="Executor.",
                code_execution_config={"use_docker": False},
                human_input_mode="NEVER",
                default_auto_reply="...",
            )

            researcher = autogen.AssistantAgent(
                name="Researcher",
                llm_config=self.llm_config,
                system_message="You are a Helpful Assistant. Answer the user question using search_documents tool. Answer in Polish."
            )

            critic = autogen.AssistantAgent(
                name="Critic",
                llm_config=self.llm_config,
                system_message="Check the answer. If good, say 'TERMINATE'."
            )

            autogen.register_function(
                search_documents,
                caller=researcher,
                executor=user_proxy,
                name="search_documents",
                description="Search documents"
            )

            # 4. Chat - ZMIANA NA ROUND_ROBIN
            # To wymusi: Admin -> Researcher -> Critic -> Admin ...
            # Dzięki temu Researcher NIE ZOSTANIE POMINIĘTY.
            groupchat = autogen.GroupChat(
                agents=[user_proxy, researcher, critic], 
                messages=[], 
                max_round=10,
                speaker_selection_method="round_robin" 
            )
            manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)

            user_proxy.initiate_chat(
                manager,
                message=f"Pytanie użytkownika: '{user_query}'. Znajdź odpowiedź w dokumentach i odpowiedz po polsku."
            )

            # 5. Wyciąganie odpowiedzi (Pancerne)
            final_response = "Przepraszam, system nie wygenerował odpowiedzi (Brak wiadomości Researchera)."
            
            # DEBUG: Wypisz historię w logach
            logger.info(f"📊 Chat History Count: {len(groupchat.messages)}")
            
            for msg in reversed(groupchat.messages):
                role = msg.get('name')
                content = msg.get('content')
                
                # Szukamy cokolwiek co powiedział Researcher
                if role == "Researcher" and content:
                    final_response = content
                    break
            
            # OSTATECZNE ZABEZPIECZENIE
            if final_response is None:
                return "CRITICAL ERROR: Logic returned None."
                
            return final_response

        except Exception as e:
            logger.error(f"❌ Agentic Workflow Exception: {e}")
            return f"System Error: {str(e)}"