import autogen
from langfuse import Langfuse

def get_reviewer_agent(llm_config: dict) -> autogen.AssistantAgent:
    """
    Tworzy i konfiguruje agenta Krytyka (Reviewer/Critic).
    Pobiera prompt dynamicznie z Langfuse.
    """
    langfuse = Langfuse()
    
    # Pobieramy prompt z chmury (z cache)
    # Dzięki Lazy Loading w main app, tutaj import jest bezpieczny
    critic_prompt = langfuse.get_prompt("synapse-critic").compile()

    critic = autogen.AssistantAgent(
        name="Critic",
        llm_config=llm_config,
        system_message=critic_prompt
    )
    
    return critic