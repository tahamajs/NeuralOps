"""
Unified LLM client supporting OpenAI, Anthropic, and local models via LiteLLM.
"""
import os, logging
from django.conf import settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = getattr(settings, 'LLM_MODEL', 'gpt-4o-mini')
DEFAULT_PROVIDER = getattr(settings, 'LLM_PROVIDER', 'openai')


def get_llm_response(messages: list, model: str = None, stream: bool = False):
    """
    Send messages to the configured LLM and return the response.
    Supports OpenAI, Anthropic, and LiteLLM.
    """
    provider = DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL

    if provider == 'anthropic':
        return _call_anthropic(messages, model, stream)
    elif provider == 'litellm':
        return _call_litellm(messages, model, stream)
    else:
        return _call_openai(messages, model, stream)


def _call_openai(messages, model, stream):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
    )
    if stream:
        return _stream_openai(response)
    return response.choices[0].message.content


def _call_anthropic(messages, model, stream):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY', ''))
    # Convert messages to Anthropic format
    system = None
    conv = []
    for m in messages:
        if m['role'] == 'system':
            system = m['content']
        else:
            conv.append({'role': m['role'], 'content': m['content']})
    kwargs = {'model': model, 'messages': conv, 'max_tokens': 4096}
    if system:
        kwargs['system'] = system
    if stream:
        return _stream_anthropic(client, **kwargs)
    response = client.messages.create(**kwargs)
    return response.content[0].text


def _call_litellm(messages, model, stream):
    import litellm
    response = litellm.completion(model=model, messages=messages, stream=stream)
    if stream:
        return _stream_litellm(response)
    return response.choices[0].message.content


def _stream_openai(response):
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _stream_anthropic(client, **kwargs):
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


def _stream_litellm(response):
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate_code(prompt: str, language: str = 'python', context: dict = None) -> str:
    """
    Generate code based on a prompt with optional context.
    """
    system = f"You are an expert {language} developer. Write clean, well-documented code. "
    if context:
        system += f"Context: {context}"

    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': prompt},
    ]
    return get_llm_response(messages)


def review_code(code: str, language: str = 'python') -> dict:
    """
    Review code and return issues and suggestions.
    """
    system = f"You are an expert {language} code reviewer. Analyze the code and return a JSON object with 'issues' (list of problems), 'suggestions' (list of improvements), and 'overall_assessment' (string)."
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': code},
    ]
    response = get_llm_response(messages)
    import json
    try:
        return json.loads(response)
    except:
        return {'issues': [], 'suggestions': [], 'overall_assessment': response}


def generate_tests(code: str, language: str = 'python', framework: str = 'pytest') -> str:
    """
    Generate unit tests for the provided code.
    """
    system = f"You are an expert {language} tester. Write comprehensive {framework} tests for the code below."
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': code},
    ]
    return get_llm_response(messages)