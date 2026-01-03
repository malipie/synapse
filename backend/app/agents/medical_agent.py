import logging
import os
import json
from typing import Dict, Any, List, Annotated

import autogen
from app.rag.vector_store import VectorStore
from app.core.config import settings

logger = logging.getLogger(__name__)

class MedicalAgentTeam:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        self.llm_config = {
            "config_list": [{
                "model": settings.OPENAI_MODEL_NAME if hasattr(settings, "OPENAI_MODEL_NAME") else "gpt-3.5-turbo",
                "api_key": settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"),
            }],
            "temperature": 0.1, 
        }

    def run(self, user_query: str) -> str:
        try:
            # 1. Definicja Narzędzia (Czysta i Pancerna)
            def search_documents(
                query: Annotated[str, "Keywords to search. Do NOT use full sentences."]
            ) -> str:
                print(f"🔎 DEBUG: Researcher is searching for: '{query}'")
                try:
                    results = self.vector_store.search(query) # Bez 'k', baza ma domyślny limit
                    
                    if not results:
                        return "No documents found containing these keywords."
                    
                    formatted_results = []
                    for i, res in enumerate(results):
                        # Ekstrakcja treści (bezpieczna)
                        content = None
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

                        # Czyszczenie treści (usuwamy nadmiar spacji i kresek)
                        clean_content = " ".join(content.split())[:2000] # Limit znaków na fragment
                        
                        # Ekstrakcja nazwy pliku
                        filename = "Unknown Source"
                        if isinstance(res, dict):
                            meta = res.get('metadata') or res.get('payload') or {}
                            filename = meta.get('filename') or meta.get('source') or "Unknown"
                        
                        formatted_results.append(f"Source: {filename}\nContent: {clean_content}")

                    return "\n\n".join(formatted_results)

                except Exception as e:
                    return f"Search Error: {str(e)}"

            # 2. Agenci
            user_proxy = autogen.UserProxyAgent(
                name="Admin",
                system_message="Executor. After tool execution, output the result so Researcher can analyze it.",
                code_execution_config={"use_docker": False},
                human_input_mode="NEVER",
                default_auto_reply="...",
            )

            researcher = autogen.AssistantAgent(
                name="Researcher",
                llm_config=self.llm_config,
                system_message=(
                    "You are a Senior Medical Regulatory Analyst. "
                    "Your goal is to answer the user's question based ONLY on the search results. "
                    "1. Search for keywords.\n"
                    "2. ANALYZE the results provided by the Admin.\n"
                    "3. Write a summary answering the user's question.\n"
                    "4. If the user asks in Polish, ANSWER IN POLISH.\n"
                )
            )

            critic = autogen.AssistantAgent(
                name="Critic",
                llm_config=self.llm_config,
                system_message=(
                    "You are a Quality Assurance Auditor. "
                    "Check the Researcher's answer. "
                    "If it answers the question and is in the correct language, say 'TERMINATE'. "
                    "Otherwise, tell them what to fix."
                )
            )

            # 3. Rejestracja
            autogen.register_function(
                search_documents,
                caller=researcher,
                executor=user_proxy,
                name="search_documents",
                description="Search for medical documents."
            )

            # 4. Chat Manager
            groupchat = autogen.GroupChat(
                agents=[user_proxy, researcher, critic], 
                messages=[], 
                max_round=15, # Dajemy im więcej czasu
                speaker_selection_method="auto"
            )
            manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)

            logger.info(f"Starting Agentic Workflow for query: {user_query}")
            
            user_proxy.initiate_chat(
                manager,
                message=f"User Question: '{user_query}'. Find answer in documents. Respond in Polish."
            )

            # 5. Wyciąganie odpowiedzi (NAPRAWIONE)
            final_response = "Przepraszam, nie udało się wygenerować odpowiedzi."
            
            # Szukamy od końca
            for msg in reversed(groupchat.messages):
                # Ignorujemy Admina (to tylko logi z narzędzi)
                if msg['name'] == "Admin":
                    continue
                
                content = msg.get('content', '')
                if not content:
                    continue
                
                # Jeśli trafimy na TERMINATE, bierzemy treść ale usuwamy słowo kluczowe
                if "TERMINATE" in content:
                    clean_text = content.replace("TERMINATE", "").strip()
                    if clean_text: # Jeśli po usunięciu coś zostało (np. "Oto odpowiedź. TERMINATE")
                        final_response = clean_text
                        break
                    else:
                        continue # Jeśli zostało samo puste pole, szukamy dalej w tył
                
                # Jeśli to normalna wiadomość od Badacza/Krytyka
                final_response = content
                break
            
            return final_response

        except Exception as e:
            logger.error(f"Agentic Workflow failed: {e}")
            return f"System Error: {str(e)}"