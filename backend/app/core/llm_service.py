import logging
import os
from typing import List, Dict, Optional

from litellm import completion
import litellm
from langfuse import Langfuse

# --- Configuration ---
# Set LiteLLM callbacks for Langfuse observability
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

import phonenumbers
from presidio_analyzer import (
    AnalyzerEngine, 
    RecognizerRegistry, 
    EntityRecognizer, 
    PatternRecognizer,
    Pattern,
    RecognizerResult
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Custom Recognizers ---

class GooglePhoneRecognizer(EntityRecognizer):
    """
    Custom recognizer for Polish phone numbers using the Google phonenumbers library.
    """
    def __init__(self, supported_language: str = "pl", default_region: str = "PL"):
        super().__init__(supported_entities=["PHONE_NUMBER"], supported_language=supported_language)
        self.default_region = default_region
    
    def load(self) -> None: 
        pass
    
    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results = []
        try:
            matcher = phonenumbers.PhoneNumberMatcher(text, self.default_region)
            for match in matcher:
                results.append(RecognizerResult("PHONE_NUMBER", match.start, match.end, 1.0))
        except Exception:
            pass # Fail silently for phone parsing errors
        return results

class PeselRecognizer(PatternRecognizer):
    """
    Regex-based recognizer for Polish PESEL numbers (Citizen ID).
    """
    def __init__(self):
        # Basic regex for 11 digits
        patterns = [Pattern(name="pesel_pattern", regex=r"\b\d{11}\b", score=0.8)]
        super().__init__(supported_entity="PESEL", patterns=patterns, supported_language="pl")

class SecureLLMService:
    """
    Service responsible for PII sanitization and LLM interaction.
    """
    def __init__(self):
        logger.info("Initializing SecureLLMService (Heavy Load)...")
        
        # --- PII Engine Initialization (Presidio) ---
        
        # Configuration to suppress warnings and map Spacy Polish tags to Presidio entities
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "pl", "model_name": "pl_core_news_md"}, 
                {"lang_code": "en", "model_name": "en_core_web_lg"}
            ],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": {
                    "pl_core_news_md": {
                        "persName": "PERSON",
                        "placeName": "LOCATION",
                        "orgName": "ORGANIZATION",
                        "PERSON": "PERSON", # Fallbacks
                        "LOC": "LOCATION",
                        "ORG": "ORGANIZATION"
                    }
                },
                "low_score_entity_names": [], # Suppress warning
                "labels_to_ignore": [],       # Suppress warning
            }
        }
        
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # Load recognizers
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(languages=["pl", "en"])
        registry.add_recognizer(GooglePhoneRecognizer(default_region="PL"))
        registry.add_recognizer(PeselRecognizer())
        
        self.analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)
        self.anonymizer = AnonymizerEngine()
        
        # Determine model from settings or env
        self.model_name = settings.OPENAI_MODEL_NAME if hasattr(settings, "OPENAI_MODEL_NAME") else "gpt-3.5-turbo"
        self.langfuse = Langfuse()

    def _sanitize_input(self, text: str) -> str:
        """
        Detects and masks PII data in the input text.
        """
        try:
            results = self.analyzer.analyze(
                text=text, 
                language="pl",
                entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "NIP", "PESEL", "CREDIT_CARD", "LOCATION"]
            )
            anonymized = self.anonymizer.anonymize(
                text=text, 
                analyzer_results=results,
                operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<PII_REDACTED>"})}
            )
            return anonymized.text
        except Exception as e:
            logger.error(f"PII Masking failed: {e}")
            # Fallback: return original text to ensure system continuity
            return text

    async def get_chat_response(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Generates a chat response using LiteLLM, including PII sanitization for user input.
        """
        if os.getenv("MOCK_LLM") == "true":
            return "This is a simulated CHAT response (Mock Mode)"
        try:
            # Fetch prompt from Langfuse
            system_prompt = self.langfuse.get_prompt("synapse-smalltalk").compile()
            
            # Inject system prompt
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                messages[0]["content"] = system_prompt

            # Sanitize the last user message
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] = self._sanitize_input(messages[-1]["content"])

            response = completion(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                metadata={
                    "tags": ["small-talk"],
                    "generation_name": "small-talk-response", 
                    "trace_user_id": "user-synapse"
                } 
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Chat Error: {e}")
            raise e

    async def classify_intent(self, query: str) -> str:
        """
        Classifies user intent (CHAT vs RAG) using a zero-shot router prompt.
        """
        if os.getenv("MOCK_LLM") == "true":
            return "RAG"
        try:
            safe_query = self._sanitize_input(query)
            router_prompt = self.langfuse.get_prompt("synapse-router")
            
            response = completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": router_prompt.compile()},
                    {"role": "user", "content": safe_query}
                ],
                temperature=0.0,
                max_tokens=10,
                metadata={
                    "tags": ["router"],
                    "generation_name": "intent-classification"
                }
            )
            intent = response.choices[0].message.content.strip().upper()
            return "RAG" if "RAG" in intent else "CHAT"
        except Exception as e:
            logger.error(f"Router failed: {e}")
            # Default to RAG in case of router failure to be safe
            return "RAG"

# --- SINGLETON PATTERN ---
_instance = None

def get_secure_llm():
    global _instance
    if _instance is None:
        _instance = SecureLLMService()
    return _instance