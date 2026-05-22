import logging
import json
from celery import shared_task
from django.utils import timezone
from .models import SupportTicket, TicketMessage, Solution

logger = logging.getLogger(__name__)


def _call_llm(messages, tools=None):
    """Call the OpenAI LLM (or another provider)."""
    import os
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        temperature=0.3,
    )
    return response


def _search_web(query, max_results=5):
    """Search the web using DuckDuckGo (free, no API key)."""
    from duckduckgo_search import DDGS
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    'title': r['title'],
                    'url': r['href'],
                    'snippet': r['body'],
                })
    except Exception as e:
        logger.warning("Web search failed: %s", e)
    return results


@shared_task(bind=True, max_retries=3)
def process_ticket_task(self, ticket_id):
    """
    Process a new support ticket: generate summary, steps, and search the web.
    """
    try:
        ticket = SupportTicket.objects.get(id=ticket_id)
    except SupportTicket.DoesNotExist:
        return

    ticket.status = 'analyzing'
    ticket.save(update_fields=['status'])

    # Build prompt
    prompt = f"""You are an expert AI research assistant. A researcher submitted the following problem:

Title: {ticket.title}
Description: {ticket.description}
Code Snippet: {ticket.code_snippet or 'None'}
Tags: {json.dumps(ticket.tags)}

Please provide:
1. A concise summary of the problem.
2. A step-by-step solution approach.
3. If applicable, a mathematical explanation (use LaTeX syntax).

Return your response as JSON with keys: summary, steps (list of strings), math_explanation (string)."""

    try:
        response = _call_llm([{"role": "user", "content": prompt}])
        content = response['choices'][0]['message']['content']
        # Parse JSON
        data = json.loads(content)
        ticket.ai_summary = data.get('summary', '')
        ticket.ai_steps = data.get('steps', [])
        ticket.status = 'solved'
        ticket.save(update_fields=['ai_summary', 'ai_steps', 'status'])

        # Save AI message
        TicketMessage.objects.create(
            ticket=ticket,
            role='assistant',
            content=f"Summary: {ticket.ai_summary}\n\nSteps:\n" + "\n".join(f"- {s}" for s in ticket.ai_steps),
        )

        # Search the web for additional resources
        results = _search_web(ticket.title)
        ticket.web_sources = results
        ticket.save(update_fields=['web_sources'])

        if results:
            web_msg = "Related resources:\n" + "\n".join(
                f"- [{r['title']}]({r['url']})" for r in results
            )
            TicketMessage.objects.create(ticket=ticket, role='assistant', content=web_msg)

    except Exception as e:
        logger.exception("Ticket processing failed")
        ticket.status = 'open'
        ticket.save(update_fields=['status'])
        TicketMessage.objects.create(
            ticket=ticket,
            role='system',
            content=f"AI processing failed: {str(e)}",
        )


@shared_task(bind=True, max_retries=3)
def process_ticket_message_task(self, ticket_id, user_message):
    """
    Process a follow‑up message in a ticket and generate a response.
    """
    try:
        ticket = SupportTicket.objects.get(id=ticket_id)
    except SupportTicket.DoesNotExist:
        return

    # Build conversation history
    messages = [{"role": "system", "content": "You are an expert AI research assistant."}]
    for msg in ticket.messages.order_by('created_at'):
        role = "user" if msg.role == "user" else "assistant"
        messages.append({"role": role, "content": msg.content})

    try:
        response = _call_llm(messages)
        ai_content = response['choices'][0]['message']['content']

        # Save AI reply
        TicketMessage.objects.create(ticket=ticket, role='assistant', content=ai_content)

        # Send via WebSocket (using channel layer)
        try:
            from channels.layers import get_channel_layer
            from asgiref.sync import async_to_sync
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                f'support_ticket_{ticket_id}',
                {
                    'type': 'chat_message',
                    'role': 'assistant',
                    'content': ai_content,
                }
            )
        except Exception:
            pass

    except Exception as e:
        logger.exception("Message processing failed")
        TicketMessage.objects.create(
            ticket=ticket,
            role='system',
            content=f"AI response failed: {str(e)}",
        )