"""
LangGraph research agent graph – processes user queries, retrieves context,
calls tools, and generates responses.
"""

import json
import logging
from typing import Annotated, List

from django.conf import settings
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)


# State
class ResearchState(TypedDict):
    messages: Annotated[List, add_messages]
    user_id: int
    retrieved_context: List[str]
    final_answer: str


llm = ChatOpenAI(
    model=getattr(settings, "OPENAI_MODEL", "gpt-4o"),
    temperature=0.3,
    api_key=settings.OPENAI_API_KEY or "mock-key-for-import",
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = """You are a helpful research assistant and coach for the NeuralOps platform.
You have access to the user's experiments, notes, saved papers, reputation, and collaborator matches.

Your job:
- Answer questions about their research.
- Suggest new experiments or improvements based on their history.
- Provide motivational feedback about their progress.
- Recommend collaborators from their matches.
- Generate research ideas and next steps.

Always be encouraging and data‑driven. Use the tools to fetch real data when needed."""


def agent_node(state: ResearchState):
    """Call the LLM with tools."""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: ResearchState):
    """Execute tool calls."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_func = next((t for t in ALL_TOOLS if t.name == tool_name), None)
            if tool_func:
                try:
                    result = tool_func.invoke(tool_args)
                    tool_messages.append(
                        ToolMessage(content=json.dumps(result), tool_call_id=tc["id"])
                    )
                except Exception as e:
                    tool_messages.append(
                        ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"])
                    )
            else:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool {tool_name} not found", tool_call_id=tc["id"]
                    )
                )
    return {"messages": tool_messages}


def should_continue(state: ResearchState):
    """Decide whether to continue to tools or end."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()
