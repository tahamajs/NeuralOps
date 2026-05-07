import os, json, time, requests
from django.utils import timezone
from .models import AgentSession, AgentMemory, AgentAction
from .tools import ToolRegistry
from .memory import TwoTierMemory

class AgentRuntime:
    """Main agent loop. Call from Celery or a management command."""
    
    def __init__(self, session: AgentSession):
        self.session = session
        self.tools = ToolRegistry.get_tools(session.agent_type)
        self.memory = TwoTierMemory(session)
        
        # LLM API base (use any provider – here Anthropic as example)
        self.llm_api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        self.llm_model = 'claude-sonnet-4-20250514'
    
    def run_cycle(self) -> dict:
        """Execute one agent cycle."""
        cycle = self.session.cycles_completed + 1
        
        # Step 1: Build prompt with memory
        prompt = self._build_prompt()
        
        # Step 2: Call LLM to decide action
        response = self._call_llm(prompt)
        
        # Step 3: Parse and execute tool calls
        for action in self._parse_actions(response):
            tool_fn = self.tools.get(action['tool'])
            if tool_fn:
                result = tool_fn(action['params'], self.session)
                AgentAction.objects.create(
                    session=self.session,
                    cycle=cycle,
                    action_type='tool_call',
                    tool_name=action['tool'],
                    input_data=action['params'],
                    output_summary=json.dumps(result)[:2000],
                    llm_tokens_used=response.get('usage', {}).get('input_tokens', 0) // len(self._parse_actions(response)),
                    success=result.get('status') != 'error',
                )
        
        # Step 4: Update memory
        self.memory.update_working_memory(response.get('content', [{}])[0].get('text', '')[:3000])
        self.memory.compress_if_needed()
        
        # Step 5: Increment cycle
        self.session.cycles_completed = cycle
        self.session.save(update_fields=['cycles_completed', 'updated_at'])
        
        return {'cycle': cycle, 'status': 'completed'}
    
    def _build_prompt(self) -> list:
        """Build LLM prompt with system instructions, memory, and tool definitions."""
        system_prompt = f"""You are an autonomous ML research agent running inside NeuralOps.
Your objective: {self.session.objective}
Target metric: {self.session.metric_name} ({self.session.metric_direction})
Cycles completed: {self.session.cycles_completed}/{self.session.max_cycles}
Budget remaining: ${(self.session.max_llm_cost_cents - self.session.total_llm_cost_cents)/100:.2f}

You have access to these tools: {', '.join(self.tools.keys())}

When you propose a hypothesis, call `propose_hypothesis`.
When you want to run an experiment, call `run_experiment` with hyperparameters.
When you want to analyze results, call `analyze_results` with a run_id.
After analysis, propose a new hypothesis based on what you learned.

Long-term memory summary: {self.memory.long_term_summary or 'No prior knowledge yet.'}
Current working context: {self.memory.working_memory or 'No current context.'}
"""
        
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': 'Execute the next cycle. Analyze previous results if available, propose a hypothesis, and run an experiment.'},
        ]
    
    def _call_llm(self, messages: list) -> dict:
        """Call the LLM API with tool definitions."""
        headers = {
            'x-api-key': self.llm_api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }
        
        tool_defs = [
            {
                'name': name,
                'description': fn.__doc__ or f'Tool: {name}',
                'input_schema': {
                    'type': 'object',
                    'properties': {'params': {'type': 'object'}},
                    'required': ['params'],
                },
            }
            for name, fn in self.tools.items()
        ]
        
        body = {
            'model': self.llm_model,
            'max_tokens': 1024,
            'messages': messages,
            'tools': tool_defs,
        }
        
        try:
            resp = requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=body, timeout=30)
            data = resp.json()
            
            # Track cost (~$3 per 1M input tokens, ~$15 per 1M output tokens)
            usage = data.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            cost_cents = int((input_tokens * 3 + output_tokens * 15) / 10000)
            self.session.total_llm_cost_cents += cost_cents
            self.session.save(update_fields=['total_llm_cost_cents'])
            
            return data
        except Exception as e:
            return {'error': str(e), 'content': []}
    
    def _parse_actions(self, response: dict) -> list:
        """Extract tool calls from LLM response."""
        actions = []
        for block in response.get('content', []):
            if block.get('type') == 'tool_use':
                actions.append({
                    'tool': block.get('name', ''),
                    'params': block.get('input', {}).get('params', block.get('input', {})),
                })
        # If LLM didn't use tools but gave text, treat first sentence as a hypothesis
        if not actions:
            text = response.get('content', [{}])[0].get('text', '')
            if text:
                actions.append({'tool': 'propose_hypothesis', 'params': {'hypothesis': text[:2000]}})
        return actions


def run_agent_loop(session_id: str, max_cycles: int = None):
    """Entry point for Celery task. Runs agent cycles until budget exhausted."""
    session = AgentSession.objects.get(id=session_id)
    
    if session.status not in ('active',):
        return {'error': f'Session not active (status: {session.status})'}
    
    runtime = AgentRuntime(session)
    cycles_to_run = min(max_cycles or session.max_cycles - session.cycles_completed, session.max_cycles)
    
    for _ in range(cycles_to_run):
        session.refresh_from_db()
        if session.status != 'active':
            break
        if session.total_llm_cost_cents >= session.max_llm_cost_cents:
            session.status = 'review'
            session.save(update_fields=['status'])
            break
        runtime.run_cycle()
        time.sleep(2)  # Short pause between cycles
    
    return {'cycles_completed': session.cycles_completed, 'status': session.status}