"""
LLM-powered log analyzer using the SemiSMAC approach.
Uses an LLM to parse experiment logs, detect anomalies, and suggest fixes.
"""
import json, logging
import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class LLMLogAnalyzer:
    """
    Uses an LLM (Anthropic Claude, OpenAI GPT, or local model) to analyze
    experiment logs and detect anomalies with suggested fixes.
    """
    def __init__(self):
        self.model = getattr(settings, 'LLM_MODEL', 'claude-sonnet-4-20250514')
        self.provider = getattr(settings, 'LLM_PROVIDER', 'anthropic')
        self.api_key = getattr(settings, 'ANTHROPIC_API_KEY', '') or getattr(settings, 'OPENAI_API_KEY', '')

    def analyze_logs(self, logs: str, metrics: dict = None) -> list:
        """
        Analyze a batch of experiment logs and return a list of detected anomalies.
        Each anomaly is a dict with: type, severity, description, suggested_fix, step.
        """
        if not logs.strip():
            return []

        # Build prompt
        prompt = self._build_prompt(logs, metrics)
        response = self._call_llm(prompt)

        if not response:
            return []

        # Parse the LLM response into structured anomalies
        return self._parse_response(response)

    def _build_prompt(self, logs: str, metrics: dict = None) -> str:
        metrics_str = ""
        if metrics:
            metrics_str = json.dumps(metrics, indent=2)

        return f"""You are an expert ML engineer analyzing experiment logs.
Detect anomalies, diagnose issues, and suggest concrete fixes.

ANOMALY TYPES:
- loss_spike: Sudden large increase in loss
- loss_plateau: Loss stops improving for many steps
- divergence: Loss grows instead of decreasing
- gradient_explosion: Gradients become very large
- gradient_vanishing: Gradients become near zero
- memory_leak: GPU/CPU memory usage steadily increases
- slow_convergence: Loss decreases but very slowly
- overfitting: Train loss improves but eval loss worsens
- underfitting: Both train and eval loss remain high
- hardware_issue: GPU errors, temperature issues, etc.
- data_corruption: Data loading errors, NaN values

EXPERIMENT LOGS:
{logs[-5000:]}

METRICS (latest):
{metrics_str}

Return a JSON list of anomalies. Each entry must have:
{{
  "type": "<anomaly_type>",
  "severity": <0.0 to 1.0>,
  "description": "<detailed description>",
  "suggested_fix": "<concrete action to fix>",
  "step": <step_number_or_null>
}}

Return ONLY valid JSON array. No other text.
"""

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        try:
            if self.provider == 'anthropic':
                return self._call_anthropic(prompt)
            elif self.provider == 'openai':
                return self._call_openai(prompt)
            else:
                logger.warning("Unknown LLM provider: %s", self.provider)
                return None
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

    def _call_anthropic(self, prompt: str) -> str:
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }
        body = {
            'model': self.model,
            'max_tokens': 1024,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        resp = requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=body, timeout=30)
        data = resp.json()
        return data.get('content', [{}])[0].get('text', '')

    def _call_openai(self, prompt: str) -> str:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        body = {
            'model': self.model,
            'max_tokens': 1024,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        resp = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=body, timeout=30)
        data = resp.json()
        return data['choices'][0]['message']['content']

    def _parse_response(self, response: str) -> list:
        """Parse the LLM's JSON response into a list of anomaly dicts."""
        # Extract JSON from the response (handle markdown fences)
        text = response.strip()
        if text.startswith('```'):
            # Remove code fences
            lines = text.split('\n')
            text = '\n'.join(lines[1:-1]) if len(lines) > 2 else text

        try:
            anomalies = json.loads(text)
            if isinstance(anomalies, dict):
                anomalies = [anomalies]
            return anomalies if isinstance(anomalies, list) else []
        except json.JSONDecodeError:
            # Try to extract JSON array from text
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return []