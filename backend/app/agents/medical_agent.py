import logging
import os
import autogen
from langfuse import Langfuse
import litellm 

from app.rag.vector_store import VectorStore
from app.core.config import settings

# Import custom modules
from app.agents.tools import get_search_tool
from app.agents.reviewer_agent import get_reviewer_agent

logger = logging.getLogger(__name__)

class MedicalAgentTeam:
    """
    Manages the multi-agent workflow for medical RAG tasks.
    """
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.langfuse = Langfuse()
        
        # LLM Configuration
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
            # Enable Observability callbacks
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["langfuse"]
            
            # 1. Fetch Researcher Prompt from Langfuse
            researcher_prompt = self.langfuse.get_prompt("synapse-researcher").compile()

            # 2. Setup Admin Agent (Executor)
            user_proxy = autogen.UserProxyAgent(
                name="Admin",
                system_message="Executor.",
                code_execution_config={"use_docker": False},
                human_input_mode="NEVER",
                default_auto_reply="...",
            )

            # 3. Setup Researcher Agent
            researcher = autogen.AssistantAgent(
                name="Researcher",
                llm_config=self.llm_config,
                system_message=researcher_prompt
            )

            # 4. Setup Reviewer/Critic Agent (from external module)
            critic = get_reviewer_agent(self.llm_config)

            # 5. Register Tools
            search_tool_func = get_search_tool(self.vector_store)

            autogen.register_function(
                search_tool_func,
                caller=researcher,
                executor=user_proxy,
                name="search_documents",
                description="Search for medical documents based on keywords."
            )

            # 6. Start Group Chat
            groupchat = autogen.GroupChat(
                agents=[user_proxy, researcher, critic], 
                messages=[], 
                max_round=10,
                speaker_selection_method="round_robin" 
            )
            manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)

            # Initiate chat with specific instructions
            # Note: We specifically ask for the response in Polish for the end user.
            user_proxy.initiate_chat(
                manager,
                message=f"User Query: '{user_query}'. Find the answer in documents and reply in Polish."
            )

            # 7. Extract Final Result
            final_response = "Przepraszam, system nie wygenerował odpowiedzi." # Default fallback in Polish
            for msg in reversed(groupchat.messages):
                if msg.get('name') == "Researcher" and msg.get('content'):
                    final_response = msg.get('content')
                    break
            
            return final_response

        except Exception as e:
            logger.error(f"❌ Agentic Workflow Exception: {e}")
            return f"System Error: {str(e)}"