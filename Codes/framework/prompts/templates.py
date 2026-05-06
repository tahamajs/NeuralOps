"""Prompt template engine – simple strings and chat conversations."""

import re
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class PromptTemplate:
    """
    A simple prompt template with variable interpolation.

    Variables are marked with double‑braces: {{variable_name}}.
    Defaults can be provided inline: {{variable_name:default_value}}.

    Usage:
        t = PromptTemplate("You are a {{role}}. Answer: {{question}}")
        result = t.render(role="helpful assistant", question="What is ML?")
    """
    template: str
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Extract variable names from the template
        self._var_names = re.findall(r'\{\{(\w+)(?::[^}]*)?\}\}', self.template)

    def render(self, **kwargs) -> str:
        """Render the template by substituting all provided values."""
        merged = {**self.variables, **kwargs}
        result = self.template
        for var in self._var_names:
            if var in merged:
                val = str(merged[var])
                result = re.sub(r'\{\{' + var + r'(?::[^}]*)?\}\}', val, result)
        # Apply defaults for missing variables
        result = re.sub(r'\{\{(\w+):([^}]*)\}\}', r'\2', result)
        return result

    def to_dict(self) -> dict:
        return {"type": "simple", "template": self.template,
                "variables": self.variables, "metadata": self.metadata}


@dataclass
class ChatPromptTemplate:
    """
    A conversation‑style prompt template.  Each message has a role
    ('system', 'user', 'assistant') and content with optional variables.

    Usage:
        t = ChatPromptTemplate(messages=[
            {"role": "system", "content": "You are {{role}}."},
            {"role": "user",   "content": "{{question}}"},
        ])
        msgs = t.render(role="assistant", question="Hi")
    """
    messages: List[Dict[str, str]]
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs) -> List[Dict[str, str]]:
        merged = {**self.variables, **kwargs}
        output = []
        for msg in self.messages:
            content = msg["content"]
            for var, val in merged.items():
                content = content.replace("{{" + var + "}}", str(val))
            output.append({"role": msg["role"], "content": content})
        return output

    def to_dict(self) -> dict:
        return {"type": "chat", "messages": self.messages,
                "variables": self.variables, "metadata": self.metadata}