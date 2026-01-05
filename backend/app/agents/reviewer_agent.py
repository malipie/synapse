import autogen
from langfuse import Langfuse

def get_reviewer_agent(llm_config: dict) -> autogen.AssistantAgent:
    """
    Creates and configures the Critic agent.
    Dynamically fetches the prompt from Langfuse.
    """
    langfuse = Langfuse()
    
    # Fetch prompt from cloud (cached)
    # Thanks to Lazy Loading in main app, import is safe here
    critic_prompt = langfuse.get_prompt("synapse-critic").compile()

    critic = autogen.AssistantAgent(
        name="Critic",
        llm_config=llm_config,
        system_message=critic_prompt
    )
    
    return critic