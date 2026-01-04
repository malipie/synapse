import logging
import os
from typing import List, Dict, Optional

from litellm import completion
import litellm
from langfuse import Langfuse

# Config LiteLLM
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
    def __init__(self, supported_language: str = "pl", default_region: str = "PL"):
        super().__init__(supported_entities=["PHONE_NUMBER"], supported_language=supported_language)
        self.default_region = default_region
    def load(self) -> None: pass
    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results = []
        matcher = phonenumbers.PhoneNumberMatcher(text, self.default_region)
        for match in matcher:
            results.append(RecognizerResult("PHONE_NUMBER", match.start, match.end, 1.0))
        return results

class PeselRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [Pattern(name="pesel_pattern", regex=r"\b\d{11}\b", score=1.0)]
        super().__init__(supported_entity="PESEL", patterns=patterns, supported_language="pl")

class SecureLLMService:
    def __init__(self):
        logger.info("Initializing SecureLLMService (Heavy Load)...")
        
        # Init PII Engine
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "pl", "model_name": "pl_core_news_md"}, {"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(languages=["pl", "en"])
        registry.add_recognizer(GooglePhoneRecognizer(default_region="PL"))
        registry.add_recognizer(PeselRecognizer())
        
        self.analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)
        self.anonymizer = AnonymizerEngine()
        
        self.model_name = settings.OPENAI_MODEL_NAME if hasattr(settings, "OPENAI_MODEL_NAME") else "gpt-3.5-turbo"
        self.langfuse = Langfuse()

    def _sanitize_input(self, text: str) -> str:
        try:
            results = self.analyzer.analyze(
                text=text, language="pl",
                entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "NIP", "PESEL", "CREDIT_CARD"]
            )
            anonymized = self.anonymizer.anonymize(
                text=text, analyzer_results=results,
                operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<PII_REDACTED>"})}
            )
            return anonymized.text
        except Exception as e:
            logger.error(f"PII Masking failed: {e}")
            return text

    async def get_chat_response(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        try:
            system_prompt = self.langfuse.get_prompt("synapse-smalltalk").compile()
            
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                messages[0]["content"] = system_prompt

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
            return "RAG"

# --- SINGLETON PATTERN ---
_instance = None

def get_secure_llm():
    global _instance
    if _instance is None:
        _instance = SecureLLMService()
    return _instance