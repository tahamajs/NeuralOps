import logging
import re

logger = logging.getLogger(__name__)


class GuardrailEngine:
    """Apply guardrails to text input/output."""

    @classmethod
    def apply(cls, text: str, guardrail: "Guardrail") -> str:
        if guardrail.guardrail_type == "toxicity_filter":
            return cls._filter_toxicity(text, guardrail.config)
        elif guardrail.guardrail_type == "pii_redaction":
            return cls._redact_pii(text)
        elif guardrail.guardrail_type == "keyword_block":
            return cls._block_keywords(text, guardrail.config)
        elif guardrail.guardrail_type == "custom_regex":
            return cls._custom_regex(text, guardrail.config)
        return text

    @classmethod
    def apply_to_session(cls, text: str, session: "PlaygroundSession") -> str:
        """Apply all active guardrails attached to a session."""
        for attachment in session.guardrails.select_related("guardrail").filter(
            guardrail__is_active=True
        ):
            text = cls.apply(text, attachment.guardrail)
        return text

    @classmethod
    def _filter_toxicity(cls, text, config):
        # Simple keyword-based; in production, use a proper toxicity model.
        toxic_words = config.get("words", ["stupid", "idiot", "hate"])
        for word in toxic_words:
            text = text.replace(word, "[FILTERED]")
        return text

    @classmethod
    def _redact_pii(cls, text):
        # Redact emails and phone numbers using regex
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text
        )
        text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
        return text

    @classmethod
    def _block_keywords(cls, text, config):
        keywords = config.get("keywords", [])
        for kw in keywords:
            if kw.lower() in text.lower():
                return config.get("block_message", "Content blocked due to keyword.")
        return text

    @classmethod
    def _custom_regex(cls, text, config):
        pattern = config.get("pattern", "")
        replacement = config.get("replacement", "[REDACTED]")
        try:
            return re.sub(pattern, replacement, text)
        except re.error:
            return text
