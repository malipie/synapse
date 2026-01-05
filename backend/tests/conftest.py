import pytest
from unittest.mock import MagicMock, patch
# Import the CLASS, not an instantiated object
from app.core.llm_service import SecureLLMService

@pytest.fixture
def mock_dependencies():
    """
    This is the key to success. We mock all external dependencies
    BEFORE SecureLLMService attempts to use them.
    """
    with patch("app.core.llm_service.Langfuse") as MockLangfuse, \
         patch("app.core.llm_service.AnalyzerEngine") as MockAnalyzer, \
         patch("app.core.llm_service.AnonymizerEngine") as MockAnonymizer, \
         patch("app.core.llm_service.NlpEngineProvider") as MockNlpProvider, \
         patch("app.core.llm_service.completion") as MockCompletion:
        
        # Configure Mocks to return what we expect
        mock_lf_instance = MockLangfuse.return_value
        # Mock prompt compilation
        mock_lf_instance.get_prompt.return_value.compile.return_value = "System Prompt"
        
        yield {
            "langfuse": mock_lf_instance,
            "analyzer": MockAnalyzer.return_value,
            "anonymizer": MockAnonymizer.return_value,
            "completion": MockCompletion
        }

@pytest.mark.asyncio
async def test_pii_masking_logic(mock_dependencies):
    """
    We test the masking logic without loading the heavy spaCy library.
    We simulate that the Analyzer found something.
    """
    mocks = mock_dependencies
    
    # Simulate that Analyzer found a PESEL (returns a list of results)
    # In a real integration test, we would check spaCy,
    # here we check if the service correctly calls the Anonymizer.
    mocks["analyzer"].analyze.return_value = ["mock_result"]
    mocks["anonymizer"].anonymize.return_value.text = "My PESEL is <PII_REDACTED>"

    # Create an instance (LIGHTWEIGHT - because everything in __init__ is mocked)
    service = SecureLLMService()
    
    result = service._sanitize_input("My PESEL is 99999999999")
    
    assert "<PII_REDACTED>" in result
    # Check if Analyzer was called with the correct parameters
    mocks["analyzer"].analyze.assert_called()

@pytest.mark.asyncio
async def test_router_classification_rag(mock_dependencies):
    """Test the RAG router without any risky threads."""
    mocks = mock_dependencies
    
    # Simulate "RAG" response from OpenAI
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "RAG"
    mocks["completion"].return_value = mock_response

    service = SecureLLMService()
    intent = await service.classify_intent("Medical documentation")
    
    assert intent == "RAG"
    # Check if it fetched the prompt from Langfuse
    mocks["langfuse"].get_prompt.assert_called_with("synapse-router")

@pytest.mark.asyncio
async def test_router_classification_chat(mock_dependencies):
    """Test the CHAT router."""
    mocks = mock_dependencies
    
    # Simulate "CHAT" response from OpenAI
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "CHAT"
    mocks["completion"].return_value = mock_response

    service = SecureLLMService()
    intent = await service.classify_intent("Hello!")
    
    assert intent == "CHAT"