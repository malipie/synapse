import logging
from typing import List, Dict, Optional

# LiteLLM - gateway to models
from litellm import completion

# Google's libphonenumber
import phonenumbers

# Presidio - security layer (PII)
from presidio_analyzer import (
    AnalyzerEngine, 
    RecognizerRegistry, 
    EntityRecognizer, 
    RecognizerResult,
    PatternRecognizer,
    Pattern
)
# We need NlpEngineProvider to map 'pl' -> 'pl_core_news_md'
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Configuration
from app.core.config import settings

logger = logging.getLogger(__name__)

class GooglePhoneRecognizer(EntityRecognizer):
    """
    Custom Presidio Recognizer that uses Google's libphonenumber 
    to robustly detect phone numbers in any format.
    """
    
    def __init__(self, supported_language: str = "pl", default_region: str = "PL"):
        super().__init__(
            supported_entities=["PHONE_NUMBER"], 
            supported_language=supported_language
        )
        self.default_region = default_region

    def load(self) -> None:
        """No external model loading required for libphonenumber."""
        pass

    def analyze(
        self, text: str, entities: List[str], nlp_artifacts=None
    ) -> List[RecognizerResult]:
        """
        Scans text using phonenumbers.PhoneNumberMatcher.
        """
        results = []
        
        # Leniency.POSSIBLE is faster, Leniency.VALID is stricter.
        # We use the matcher which iterates over found numbers.
        matcher = phonenumbers.PhoneNumberMatcher(text, self.default_region)
        
        for match in matcher:
            result = RecognizerResult(
                entity_type="PHONE_NUMBER",
                start=match.start,
                end=match.end,
                score=1.0  # libphonenumber is highly accurate
            )
            results.append(result)
            
        return results

class SecureLLMService:
    def __init__(self):
        """
        Initializes the Secure LLM Gateway with robust PII detection for Polish.
        """
        logger.info("Initializing SecureLLMService with Google libphonenumber & Polish NLP...")
        
        # 1. Configure NLP Engine to use the Polish model
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "pl", "model_name": "pl_core_news_md"},
                {"lang_code": "en", "model_name": "en_core_web_lg"} # Fallback/Standard
            ],
        }
        
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # 2. Configuration of Presidio Registry
        registry = RecognizerRegistry()
        
        # Load standard recognizers.
        registry.load_predefined_recognizers(languages=["pl", "en"])

        # 3. Register Custom Google Phone Recognizer
        google_phone_recognizer = GooglePhoneRecognizer(default_region="PL")
        registry.add_recognizer(google_phone_recognizer)
        
        # 4. Initialize Analysis Engine with our Configured NLP Engine
        # CRITICAL FIX: Removed 'supported_languages=["pl", "en"]' argument.
        # We let the AnalyzerEngine derive supported languages from the registry automatically.
        # This prevents the "ValueError: Misconfigured engine" crash on startup.
        self.analyzer = AnalyzerEngine(
            registry=registry, 
            nlp_engine=nlp_engine
        )
        
        # Log detected languages to verify 'pl' is present
        logger.info(f"🛡️ Presidio Analyzer initialized. Supported languages: {self.analyzer.supported_languages}")

        self.anonymizer = AnonymizerEngine()
        
        # LLM Model Config
        self.model_name = settings.OPENAI_MODEL_NAME if hasattr(settings, "OPENAI_MODEL_NAME") else "gpt-3.5-turbo"

    def _sanitize_input(self, text: str) -> str:
        """
        Detects and masks PII using Presidio + libphonenumber.
        """
        try:
            # 1. Analysis (Detection)
            results = self.analyzer.analyze(
                text=text,
                language="pl", # Explicitly analyze as Polish
                entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "NIP", "PESEL", "CREDIT_CARD"]
            )
            
            # 2. Anonymization (Masking)
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={
                    "DEFAULT": OperatorConfig("replace", {"new_value": "<PII_REDACTED>"}),
                    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
                    "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
                    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
                    "PESEL": OperatorConfig("replace", {"new_value": "<PESEL>"}),
                    "NIP": OperatorConfig("replace", {"new_value": "<NIP>"}),
                    "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
                }
            )
            
            masked_text = anonymized_result.text
            
            if masked_text != text:
                logger.warning(f"🛡️ PII Detected & Masked. Original len: {len(text)}, Masked len: {len(masked_text)}")
                
            return masked_text
            
        except Exception as e:
            logger.error(f"PII Masking failed: {e}. Passing original text.")
            return text

    async def get_chat_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """
        Main entry point. Sanitize -> Route -> Generate.
        Uses LiteLLM for vendor abstraction.
        """
        try:
            # 1. Sanitize the LAST user message
            if messages and messages[-1]["role"] == "user":
                original_content = messages[-1]["content"]
                masked_content = self._sanitize_input(original_content)
                messages[-1]["content"] = masked_content

            # 2. Call LLM via LiteLLM
            logger.info(f"Sending request to LiteLLM (Model: {self.model_name})...")
            
            response = completion(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM Service Error: {e}")
            raise e
            
    async def classify_intent(self, query: str) -> str:
        """
        Router logic using LiteLLM + PII Masking.
        """
        try:
            # Mask sensitive data in the router query as well
            safe_query = self._sanitize_input(query)
            
            system_prompt = (
                "You are a routing agent for a medical system. "
                "Classify user query into: 'RAG' (documents/facts) or 'CHAT' (greeting/smalltalk). "
                "Return ONLY the category name."
            )
            
            response = completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": safe_query}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            intent = response.choices[0].message.content.strip().upper()
            if "RAG" in intent: return "RAG"
            return "CHAT"
            
        except Exception as e:
            logger.error(f"Router failed: {e}")
            return "RAG"

# Singleton instance
secure_llm = SecureLLMService() 