from django.core.management.base import BaseCommand
from llm_hub.models import LLMProvider, LLMModel

PROVIDERS = [
    ('OpenAI', 'https://api.openai.com/v1', 'OPENAI_API_KEY', [
        ('gpt-4o', 'GPT-4o', 128000, 4096, True, 0.005, 0.015),
        ('gpt-4o-mini', 'GPT-4o Mini', 128000, 16384, True, 0.00015, 0.0006),
        ('gpt-3.5-turbo', 'GPT-3.5 Turbo', 16385, 4096, False, 0.0005, 0.0015),
    ]),
    ('Anthropic', 'https://api.anthropic.com', 'ANTHROPIC_API_KEY', [
        ('claude-sonnet-4-20250514', 'Claude Sonnet 4', 200000, 4096, False, 0.003, 0.015),
        ('claude-opus-4-20250514', 'Claude Opus 4', 200000, 4096, False, 0.015, 0.075),
        ('claude-3-5-sonnet-latest', 'Claude 3.5 Sonnet', 200000, 4096, False, 0.003, 0.015),
    ]),
    ('Together AI', 'https://api.together.xyz/v1', 'TOGETHER_API_KEY', [
        ('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', 'Llama 3.1 70B', 131072, 4096, False, 0.00088, 0.00088),
        ('meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', 'Llama 3.1 8B', 131072, 4096, False, 0.00018, 0.00018),
    ]),
    ('Groq', 'https://api.groq.com/openai/v1', 'GROQ_API_KEY', [
        ('llama-3.1-70b-versatile', 'Llama 3.1 70B (Groq)', 131072, 4096, False, 0.00059, 0.00079),
        ('llama-3.1-8b-instant', 'Llama 3.1 8B (Groq)', 131072, 4096, False, 0.00005, 0.00008),
    ]),
    ('Cohere', '', 'COHERE_API_KEY', [
        ('command-r-plus', 'Command R+', 128000, 4096, False, 0.003, 0.015),
        ('command-r', 'Command R', 128000, 4096, False, 0.0005, 0.0015),
    ]),
    ('Google', '', 'GOOGLE_API_KEY', [
        ('gemini-2.0-flash', 'Gemini 2.0 Flash', 1048576, 8192, True, 0.0, 0.0),
        ('gemini-2.0-pro', 'Gemini 2.0 Pro', 2097152, 8192, True, 0.0, 0.0),
    ]),
    ('Mistral', 'https://api.mistral.ai/v1', 'MISTRAL_API_KEY', [
        ('mistral-large-latest', 'Mistral Large', 131072, 4096, False, 0.004, 0.012),
        ('mistral-small-latest', 'Mistral Small', 32768, 4096, False, 0.001, 0.003),
    ]),
]

class Command(BaseCommand):
    help = 'Seed LLM providers and models'

    def handle(self, *args, **options):
        for prov_name, base_url, key_env, models in PROVIDERS:
            provider, _ = LLMProvider.objects.get_or_create(
                name=prov_name,
                defaults={'api_base_url': base_url, 'api_key_env': key_env}
            )
            for model_id, display, ctx, max_tok, vision, price_in, price_out in models:
                LLMModel.objects.get_or_create(
                    provider=provider,
                    model_id=model_id,
                    defaults={
                        'display_name': display,
                        'context_window': ctx,
                        'max_output_tokens': max_tok,
                        'supports_vision': vision,
                        'pricing_input_per_1k': price_in,
                        'pricing_output_per_1k': price_out,
                    }
                )
        self.stdout.write(self.style.SUCCESS('Seeded LLM providers and models'))