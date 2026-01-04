import pytest
from unittest.mock import MagicMock, patch
# Importujemy klasę, ale jej nie uruchamiamy
from app.core.llm_service import SecureLLMService

@pytest.fixture
def mock_dependencies():
    """
    Podmieniamy wszystkie ciężkie biblioteki na atrapy (Mocki).
    Dzięki temu testy w ogóle nie dotykają spaCy ani Langfuse.
    """
    with patch("app.core.llm_service.Langfuse") as MockLangfuse, \
         patch("app.core.llm_service.AnalyzerEngine") as MockAnalyzer, \
         patch("app.core.llm_service.AnonymizerEngine") as MockAnonymizer, \
         patch("app.core.llm_service.NlpEngineProvider") as MockNlpProvider, \
         patch("app.core.llm_service.completion") as MockCompletion:
        
        # Konfiguracja Mocków
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
    """Testujemy logikę maskowania (czy woła Anonymizera)."""
    mocks = mock_dependencies
    
    # Symulujemy, że Analyzer znalazł PESEL
    mocks["analyzer"].analyze.return_value = ["mock_result"]
    mocks["anonymizer"].anonymize.return_value.text = "Mój PESEL to <PII_REDACTED>"

    # Tworzymy instancję (LEKKĄ - bo wszystko w __init__ jest zmockowane)
    service = SecureLLMService()
    
    result = service._sanitize_input("Mój PESEL to 99999999999")
    
    assert "<PII_REDACTED>" in result
    mocks["analyzer"].analyze.assert_called()

@pytest.mark.asyncio
async def test_router_classification_rag(mock_dependencies):
    mocks = mock_dependencies
    
    # Symulujemy odpowiedź "RAG" z OpenAI
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "RAG"
    mocks["completion"].return_value = mock_response

    service = SecureLLMService()
    intent = await service.classify_intent("Dokumentacja")
    
    assert intent == "RAG"
    mocks["langfuse"].get_prompt.assert_called_with("synapse-router")

@pytest.mark.asyncio
async def test_router_classification_chat(mock_dependencies):
    mocks = mock_dependencies
    
    # Symulujemy odpowiedź "CHAT" z OpenAI
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "CHAT"
    mocks["completion"].return_value = mock_response

    service = SecureLLMService()
    intent = await service.classify_intent("Cześć!")
    
    assert intent == "CHAT"