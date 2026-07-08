"""
LangGraph-based AI Research Coach agent.
Handles both background analysis and real-time chat.
"""

import json
import logging
import os
from typing import Annotated, List, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .chroma_store import search_similar_experiments

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    user_id: int
    context: dict


SYSTEM_PROMPT = """You are the NeuralOps AI Research Coach — an expert ML research assistant.

You have access to:
- The user's experiment history (via semantic search)
- The user's past results, hyperparameters, and metrics
- The ability to suggest next experiments, architecture changes, and papers to read

Be concise, specific, and data-driven. Reference actual experiment IDs and metrics when possible.
"""


@tool
def search_experiments(query: str) -> str:
    """Search the user's experiment history for relevant past runs."""
    results = search_similar_experiments(query, n_results=5)
    if not results or not results.get("documents"):
        return "No relevant experiments found."

    output = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results.get("metadatas") else {}
        output.append(f"Experiment {i+1}: {doc}\nMetadata: {json.dumps(meta)}")
    return "\n\n".join(output)


@tool
def suggest_next_experiment(run_id: str) -> str:
    """Based on a previous experiment, suggest what to try next."""
    from experiment.models import TrainingRun

    try:
        run = TrainingRun.objects.get(id=run_id)
    except TrainingRun.DoesNotExist:
        return "Experiment not found."

    config = run.config or {}
    metrics = run.final_metrics or {}
    suggestions = []

    # Check accuracy and suggest improvements
    accuracy = metrics.get("eval_accuracy", 0)
    if accuracy < 0.7:
        suggestions.append("Try a larger model or more epochs.")
    elif accuracy < 0.85:
        suggestions.append(
            "Experiment with different attention mechanisms or compression ranks."
        )
    else:
        suggestions.append(
            "Try a smaller/faster model or push the limits with harder tasks."
        )

    # Check loss trend for overfitting
    if run.training_history:
        losses = run.training_history.get("losses", [])
        if len(losses) > 5 and losses[-1] > losses[0]:
            suggestions.append(
                "Detected increasing loss — consider reducing learning rate or adding dropout."
            )

    return "Based on experiment {}: {}".format(run_id, " ".join(suggestions))


ALL_TOOLS = [search_experiments, suggest_next_experiment]


class ResearchCoachAgent:
    """LangGraph agent for the AI Research Coach."""

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.llm = ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            temperature=0.3,
        )
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._call_agent)
        workflow.add_node("tools", self._execute_tools)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", self._should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _call_agent(self, state: AgentState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _execute_tools(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        tool_results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_fn = next((t for t in ALL_TOOLS if t.name == tool_name), None)
            if tool_fn:
                result = tool_fn.invoke(tool_args)
                tool_results.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )

        return {"messages": tool_results}

    def _should_continue(self, state: AgentState):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    def chat(self, user_message: str, session_history: list = None) -> str:
        """Process a chat message and return the assistant's response."""
        messages = []
        if session_history:
            messages.extend(session_history)
        messages.append(HumanMessage(content=user_message))

        final_state = self.graph.invoke(
            {"messages": messages, "user_id": self.user_id, "context": {}}
        )
        last_ai = final_state["messages"][-1]
        return last_ai.content if isinstance(last_ai, AIMessage) else str(last_ai)
