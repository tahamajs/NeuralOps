import json, os, requests
from .tools import ToolRegistry
from .prompts import SYSTEM_PROMPT

class AgentOrchestrator:
    """Orchestrates the conversation. Calls LLM with tools, processes responses."""

    def __init__(self, session, llm_api_key: str = None):
        self.session = session
        self.llm_api_key = llm_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = "claude-sonnet-4-20250514"

    def get_history_messages(self) -> list:
        """Convert DB messages into LLM format."""
        messages = []
        for msg in self.session.messages.order_by('created_at'):
            if msg.role == 'user':
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == 'assistant':
                messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == 'tool':
                # Tool results are included in the conversation
                messages.append({"role": "user", "content": f"[Tool result: {msg.content}]"})
        return messages

    def process_user_message(self, user_text: str) -> dict:
        """Process a user message and return the agent's response."""
        # Save user message
        self.session.messages.create(role='user', content=user_text)

        # Build prompt
        system = SYSTEM_PROMPT
        history = self.get_history_messages()

        messages = [
            {"role": "system", "content": system},
        ]
        # Only include last 30 messages to stay within context
        messages.extend(history[-30:])

        # Call LLM
        response = self._call_llm(messages, ToolRegistry.get_definitions())

        # Process response
        text_response = ""
        tool_results = []

        for block in response.get("content", []):
            if block.get("type") == "text":
                text_response += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                result = ToolRegistry.execute(tool_name, tool_input)

                # Save tool result
                self.session.messages.create(role='tool', content=json.dumps(result)[:2000])

                # If it's a configure/launch, also create a rich card message
                if tool_name == "configure_experiment" and result.get("status") == "draft":
                    self.session.messages.create(
                        role='experiment_card',
                        content=json.dumps(result.get("config", {})),
                        metadata={"type": "experiment_draft", "config": result.get("config", {})}
                    )
                elif tool_name == "launch_experiment" and result.get("status") == "launched":
                    self.session.messages.create(
                        role='experiment_card',
                        content=json.dumps(result),
                        metadata={"type": "experiment_launched", "run_id": result.get("run_id")}
                    )

                tool_results.append({"tool": tool_name, "result": result})

        # Save assistant message
        if text_response:
            self.session.messages.create(role='assistant', content=text_response)

        return {
            "text": text_response,
            "tool_calls": tool_results,
        }

    def _call_llm(self, messages: list, tools: list) -> dict:
        """Call Anthropic API with tool definitions."""
        headers = {
            "x-api-key": self.llm_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "auto"},
        }
        try:
            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=45)
            return resp.json()
        except Exception as e:
            return {"content": [{"type": "text", "text": f"I'm having trouble reaching the AI backend. Error: {str(e)}"}]}