import logging
import os
import time

from .models import LLMModel, UsageRecord

logger = logging.getLogger(__name__)


class LLMGateway:
    """Unified gateway to 10+ LLM providers."""

    @classmethod
    def complete(cls, model_id: str, messages: list, **kwargs) -> dict:
        try:
            model = LLMModel.objects.select_related("provider").get(
                model_id=model_id, is_active=True
            )
        except LLMModel.DoesNotExist:
            return {"error": f"Model {model_id} not found or inactive"}

        provider_name = model.provider.name.lower()
        start = time.perf_counter()

        if provider_name == "openai":
            result = cls._call_openai(model, messages, **kwargs)
        elif provider_name == "anthropic":
            result = cls._call_anthropic(model, messages, **kwargs)
        elif provider_name == "together":
            result = cls._call_together(model, messages, **kwargs)
        elif provider_name == "groq":
            result = cls._call_groq(model, messages, **kwargs)
        elif provider_name == "cohere":
            result = cls._call_cohere(model, messages, **kwargs)
        elif provider_name == "google":
            result = cls._call_google(model, messages, **kwargs)
        elif provider_name == "mistral":
            result = cls._call_mistral(model, messages, **kwargs)
        else:
            return {"error": f"Provider {provider_name} not supported yet"}

        elapsed_ms = (time.perf_counter() - start) * 1000
        result["latency_ms"] = round(elapsed_ms, 1)
        result["cost_usd"] = cls._compute_cost(
            model, result.get("tokens_input", 0), result.get("tokens_output", 0)
        )
        return result

    @classmethod
    def _call_openai(cls, model, messages, **kwargs):
        import openai

        client = openai.OpenAI(api_key=os.environ.get(model.provider.api_key_env))
        response = client.chat.completions.create(
            model=model.model_id,
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
            tools=kwargs.get("tools"),
        )
        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "tool_calls": (
                [c.model_dump() for c in choice.message.tool_calls]
                if choice.message.tool_calls
                else []
            ),
            "tokens_input": response.usage.prompt_tokens,
            "tokens_output": response.usage.completion_tokens,
        }

    @classmethod
    def _call_anthropic(cls, model, messages, **kwargs):
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ.get(model.provider.api_key_env))
        system_msg = ""
        anthropic_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                anthropic_messages.append({"role": m["role"], "content": m["content"]})
        response = client.messages.create(
            model=model.model_id,
            messages=anthropic_messages,
            system=system_msg,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            tools=kwargs.get("tools"),
        )
        return {
            "content": response.content[0].text if response.content else "",
            "tool_calls": [],
            "tokens_input": response.usage.input_tokens,
            "tokens_output": response.usage.output_tokens,
        }

    @classmethod
    def _call_together(cls, model, messages, **kwargs):
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ.get(model.provider.api_key_env),
            base_url="https://api.together.xyz/v1",
        )
        response = client.chat.completions.create(
            model=model.model_id,
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "tool_calls": [],
            "tokens_input": response.usage.prompt_tokens,
            "tokens_output": response.usage.completion_tokens,
        }

    @classmethod
    def _call_groq(cls, model, messages, **kwargs):
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ.get(model.provider.api_key_env),
            base_url="https://api.groq.com/openai/v1",
        )
        response = client.chat.completions.create(
            model=model.model_id,
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "tool_calls": [],
            "tokens_input": response.usage.prompt_tokens,
            "tokens_output": response.usage.completion_tokens,
        }

    @classmethod
    def _call_cohere(cls, model, messages, **kwargs):
        import cohere

        co = cohere.Client(api_key=os.environ.get(model.provider.api_key_env))
        chat_history = [
            {"role": m["role"], "message": m["content"]} for m in messages[:-1]
        ]
        last_message = messages[-1]["content"]
        response = co.chat(
            model=model.model_id,
            message=last_message,
            chat_history=chat_history,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return {
            "content": response.text,
            "tool_calls": [],
            "tokens_input": response.meta.tokens.input_tokens if response.meta else 0,
            "tokens_output": response.meta.tokens.output_tokens if response.meta else 0,
        }

    @classmethod
    def _call_google(cls, model, messages, **kwargs):
        import google.generativeai as genai

        genai.configure(api_key=os.environ.get(model.provider.api_key_env))
        contents = []
        for m in messages:
            contents.append(
                {
                    "role": "user" if m["role"] == "user" else "model",
                    "parts": [m["content"]],
                }
            )
        gemini = genai.GenerativeModel(model.model_id)
        response = gemini.generate_content(contents)
        return {
            "content": response.text,
            "tool_calls": [],
            "tokens_input": response.usage_metadata.prompt_token_count,
            "tokens_output": response.usage_metadata.candidates_token_count,
        }

    @classmethod
    def _call_mistral(cls, model, messages, **kwargs):
        from mistralai import Mistral

        client = Mistral(api_key=os.environ.get(model.provider.api_key_env))
        response = client.chat.complete(
            model=model.model_id,
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return {
            "content": response.choices[0].message.content,
            "tool_calls": [],
            "tokens_input": response.usage.prompt_tokens,
            "tokens_output": response.usage.completion_tokens,
        }

    @classmethod
    def _compute_cost(cls, model, tokens_in, tokens_out) -> float:
        cost_in = (tokens_in / 1000) * float(model.pricing_input_per_1k)
        cost_out = (tokens_out / 1000) * float(model.pricing_output_per_1k)
        return round(cost_in + cost_out, 6)

    @classmethod
    def log_usage(
        cls, user, model_id, tokens_in, tokens_out, cost, source="playground"
    ):
        model = LLMModel.objects.get(model_id=model_id)
        UsageRecord.objects.create(
            user=user,
            model=model,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            cost_usd=cost,
            source=source,
        )
