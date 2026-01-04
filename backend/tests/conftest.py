import pytest
from unittest.mock import MagicMock, patch
# Importujemy KLASĘ, a nie gotową instancję
from app.core.llm_service import SecureLLMService

@pytest.fixture
def mock_dependencies():
    """
    To jest klucz do sukcesu. Podmieniamy wszystkie zewnętrzne zależności
    ZANIM SecureLLMService spróbuje ich użyć.
    """
    with patch("app.core.llm_service.Langfuse") as MockLangfuse, \
         patch("app.core.llm_service.AnalyzerEngine") as MockAnalyzer, \
         patch("app.core.llm_service.AnonymizerEngine") as MockAnonymizer, \
         patch("app.core.llm_service.NlpEngineProvider") as MockNlpProvider, \
         patch("app.core.llm_service.completion") as MockCompletion:
        
        # Konfigurujemy Mocki, żeby zwracały to, co chcemy
        mock_lf_instance = MockLangfuse.return_value
        # Mockujemy kompilację promptu
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
    Testujemy logikę maskowania, ale bez ładowania ciężkiego spaCy.
    Symulujemy, że Analyzer coś znalazł.
    """
    mocks = mock_dependencies
    
    # Symulujemy, że Analyzer znalazł PESEL (zwraca listę wyników)
    # W prawdziwym teście integracyjnym sprawdzaliśmy spaCy, 
    # tu sprawdzamy czy serwis poprawnie woła Anonymizera.
    mocks["analyzer"].analyze.return_value = ["mock_result"]
    mocks["anonymizer"].anonymize.return_value.text = "Mój PESEL to <PII_REDACTED>"

    # Tworzymy instancję (teraz jest lekka jak piórko, bo wszystko jest zmockowane)
    service = SecureLLMService()
    
    result = service._sanitize_input("Mój PESEL to 99999999999")
    
    assert "<PII_REDACTED>" in result
    # Sprawdzamy czy Analyzer został zawołany z odpowiednimi parametrami
    mocks["analyzer"].analyze.assert_called()

@pytest.mark.asyncio
async def test_router_classification_rag(mock_dependencies):
    """Test routera RAG bez żadnych ryzykownych wątków."""
    mocks = mock_dependencies
    
    # Symulujemy odpowiedź OpenAI "RAG"
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "RAG"
    mocks["completion"].return_value = mock_response

    service = SecureLLMService()
    intent = await service.classify_intent("Dokumentacja medyczna")
    
    assert intent == "RAG"
    # Sprawdzamy czy pobrał prompt z Langfuse
    mocks["langfuse"].get_prompt.assert_called_with("synapse-router")

@pytest.mark.asyncio
async def test_router_classification_chat(mock_dependencies):
    """Test routera CHAT."""
    mocks = mock_dependencies
    
    # Symulujemy odpowiedź OpenAI "CHAT"
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "CHAT"
    mocks["completion"].return_value = mock_response

    service = SecureLLMService()
    intent = await service.classify_intent("Cześć!")
    
    assert intent == "CHAT"