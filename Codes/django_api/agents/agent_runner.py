"""
Autonomous agent runner – uses AgentSession, AgentMemory, AgentAction.
This is called by Celery and runs a multi‑cycle research loop.
"""

import json

from .models import AgentAction, AgentMemory, AgentSession
from .orchestrator import get_llm_response  # we'll add a simple LLM caller
from .tools import ToolRegistry


def run_agent_cycle(session: AgentSession, user_id: int) -> dict:
    """Execute a single cycle of the autonomous agent."""
    # 1. Get memory
    memory = AgentMemory.objects.get_or_create(session=session)[0]
    # 2. Build prompt from memory + objective + tool definitions
    tool_defs = ToolRegistry.get_definitions()
    system = (
        f"You are an autonomous ML research agent. Objective: {session.objective}\n"
    )
    system += f"Previous summary: {memory.long_term_summary or 'None'}\n"
    system += "Working context: " + (memory.working_memory or "None")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Proceed with the next step. You may call tools."},
    ]
    # 3. Call LLM
    response = get_llm_response(messages, tools=tool_defs)
    # 4. Parse tool calls / actions
    for action in _parse_actions(response):
        tool_name = action["tool"]
        params = action["params"]
        result = ToolRegistry.execute(tool_name, params, user_id=user_id)
        AgentAction.objects.create(
            session=session,
            cycle=session.cycles_completed + 1,
            action_type="tool_call",
            tool_name=tool_name,
            input_data=params,
            output_summary=json.dumps(result)[:2000],
            success=result.get("status") != "error",
        )
    # 5. Update memory
    if response.get("content"):
        summary = response["content"][:3000]
        memory.working_memory = summary
        if len(memory.long_term_summary or "") + len(summary) > 2000:
            # compress long-term summary
            memory.long_term_summary = (memory.long_term_summary or "")[
                -1000:
            ] + summary
        else:
            memory.long_term_summary = (memory.long_term_summary or "") + summary
        memory.save()
    session.cycles_completed += 1
    session.save()
    return {"cycle": session.cycles_completed, "status": "running"}


def _parse_actions(response):
    """Extract tool calls from the LLM response (OpenAI function‑call format)."""
    actions = []
    # simplified parsing – adapt to actual LLM output format
    if "tool_calls" in response:
        for call in response["tool_calls"]:
            actions.append(
                {
                    "tool": call["function"]["name"],
                    "params": json.loads(call["function"]["arguments"]),
                }
            )
    return actions
