import pytest
from unittest.mock import MagicMock, patch
# Import the class, but do not instantiate it yet
from app.core.llm_service import SecureLLMService

@pytest.fixture
def mock_dependencies():
    """
    We replace all heavy libraries with dummies (Mocks).
    Thanks to this, tests do not touch spaCy or Langfuse at all.
    """
    with patch("app.core.llm_service.Langfuse") as MockLangfuse, \
         patch("app.core.llm_service.AnalyzerEngine") as MockAnalyzer, \
         patch("app.core.llm_service.AnonymizerEngine") as MockAnonymizer, \
         patch("app.core.llm_service.NlpEngineProvider") as MockNlpProvider, \
         patch("app.core.llm_service.completion") as MockCompletion:
        
        # Mock Configuration
        mock_lf_instance = MockLangfuse.return_value
        mock_lf_instance.get_prompt.return_value.compile.return_value = "System Prompt"
        
        yield {
            "langfuse": mock_lf_instance,
            "analyzer": MockAnalyzer.return_value,
            "anonymizer": MockAnonymizer.return_value,
            "completion": MockCompletion
        }

@pytest.mark.asyncio
async def test_pii_masking_logic(mock_dependencies):
    """We test the masking logic (whether it calls the Anonymizer)."""
    mocks = mock_dependencies
    
    # Simulate that Analyzer found a PESEL
    mocks["analyzer"].analyze.return_value = ["mock_result"]
    mocks["anonymizer"].anonymize.return_value.text = "My PESEL is <PII_REDACTED>"

    # Create an instance (LIGHTWEIGHT - because everything in __init__ is mocked)
    service = SecureLLMService()
    
    result = service._sanitize_input("My PESEL is 99999999999")
    
    assert "<PII_REDACTED>" in result
    mocks["analyzer"].analyze.assert_called()

@pytest.mark.asyncio
async def test_router_classification_rag(mock_dependencies):
    mocks = mock_dependencies
    
    # Simulate "RAG" response from OpenAI
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "RAG"
    mocks["completion"].return_value = mock_response

    service = SecureLLMService()
    intent = await service.classify_intent("Documentation")
    
    assert intent == "RAG"
    mocks["langfuse"].get_prompt.assert_called_with("synapse-router")

@pytest.mark.asyncio
async def test_router_classification_chat(mock_dependencies):
    mocks = mock_dependencies
    
    # Simulate "CHAT" response from OpenAI
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "CHAT"
    mocks["completion"].return_value = mock_response

    service = SecureLLMService()
    intent = await service.classify_intent("Hello!")
    
    assert intent == "CHAT"