import os
import json
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from django.conf import settings
from .tools import tool_registry_functions  # we'll refactor tools to be callable functions
from .models import ChatSession, ChatMessage
from .prompts import SYSTEM_PROMPT


# ====================== State Definition ======================
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    session_id: str
    user_id: int


# ====================== Tools (wrapped for LangChain) ======================
# We'll import the raw functions from tools.py and wrap them as LangChain tools
from .tools import (
    search_datasets,
    search_models,
    get_dataset_info,
    configure_experiment,
    launch_experiment,
    check_experiment_status,
    list_available_tasks,
    list_attention_types,
    read_latex_project,
    edit_latex_file,
    compile_latex,
)

@tool
def tool_search_datasets(query: str, limit: int = 5) -> dict:
    """Search HuggingFace for datasets matching a query."""
    return search_datasets({"query": query, "limit": limit})

@tool
def tool_search_models(query: str, task: Optional[str] = None, limit: int = 5) -> dict:
    """Search HuggingFace for models matching a query."""
    return search_models({"query": query, "task": task, "limit": limit})

@tool
def tool_get_dataset_info(dataset_id: str) -> dict:
    """Get detailed info about a specific HuggingFace dataset."""
    return get_dataset_info({"dataset_id": dataset_id})

@tool
def tool_configure_experiment(
    model_name: str,
    task_name: str,
    attention_type: str = "full",
    method: str = "baseline",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_length: int = 128,
    compression_rank: int = 64,
) -> dict:
    """Propose an experiment configuration for review."""
    return configure_experiment({
        "model_name": model_name,
        "task_name": task_name,
        "attention_type": attention_type,
        "method": method,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "compression_rank": compression_rank,
    })

@tool
def tool_launch_experiment(
    model_name: str,
    task_name: str,
    attention_type: str = "full",
    method: str = "baseline",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_length: int = 128,
    compression_rank: int = 64,
    name: Optional[str] = None,
) -> dict:
    """Launch the experiment. Only call after user confirmation."""
    return launch_experiment({
        "model_name": model_name,
        "task_name": task_name,
        "attention_type": attention_type,
        "method": method,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "compression_rank": compression_rank,
        "name": name,
    })

@tool
def tool_check_experiment_status(run_id: str) -> dict:
    """Check the status of a running/completed experiment."""
    return check_experiment_status({"run_id": run_id})

@tool
def tool_list_available_tasks() -> dict:
    """List all available GLUE benchmark tasks."""
    return list_available_tasks({})

@tool
def tool_list_attention_types() -> dict:
    """List all available attention mechanisms."""
    return list_attention_types({})

@tool
def tool_read_latex_project(project_id: str) -> dict:
    """Read a LaTeX project's files."""
    return read_latex_project({"project_id": project_id})

@tool
def tool_edit_latex_file(project_id: str, filename: str, content: str) -> dict:
    """Edit a specific file in a LaTeX project."""
    return edit_latex_file({"project_id": project_id, "filename": filename, "content": content})

@tool
def tool_compile_latex(project_id: str) -> dict:
    """Compile a LaTeX project to PDF."""
    return compile_latex({"project_id": project_id})


ALL_TOOLS = [
    tool_search_datasets,
    tool_search_models,
    tool_get_dataset_info,
    tool_configure_experiment,
    tool_launch_experiment,
    tool_check_experiment_status,
    tool_list_available_tasks,
    tool_list_attention_types,
    tool_read_latex_project,
    tool_edit_latex_file,
    tool_compile_latex,
]


# ====================== LangGraph Agent ======================
class LangGraphAgent:
    """Agent that uses LangGraph for stateful conversation with tool‑calling."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session = ChatSession.objects.get(id=session_id)

        # Initialize LLM
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            temperature=0.3,
        )

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._call_agent)
        workflow.add_node("tools", self._execute_tools)

        # Define edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")

        # Compile with memory (checkpointing)
        # In production, use SqliteSaver or PostgresSaver for persistence
        import sqlite3
        conn = sqlite3.connect(
            os.path.join(settings.BASE_DIR, "agent_checkpoints.db"), check_same_thread=False
        )
        memory = SqliteSaver(conn)
        return workflow.compile(checkpointer=memory)

    def _call_agent(self, state: AgentState) -> AgentState:
        """Node: call the LLM with the current messages, return updated state."""
        messages = state["messages"]

        # Prepend system prompt
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        response = self.llm_with_tools.invoke(messages)

        # Save assistant message to DB
        if isinstance(response, AIMessage):
            ChatMessage.objects.create(
                session=self.session,
                role='assistant',
                content=response.content or '',
            )

        return {"messages": [response]}

    def _execute_tools(self, state: AgentState) -> AgentState:
        """Node: execute any tool calls from the last AI message."""
        messages = state["messages"]
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return state

        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Find the tool function
            tool_fn = next((t for t in ALL_TOOLS if t.name == tool_name), None)
            if tool_fn:
                result = tool_fn.invoke(tool_args)
                result_str = json.dumps(result) if isinstance(result, dict) else str(result)

                # Save tool result to DB
                ChatMessage.objects.create(
                    session=self.session,
                    role='tool',
                    content=result_str[:2000],
                    metadata={"tool_name": tool_name, "tool_args": tool_args},
                )

                # If it's a configure/launch, also create rich card
                if tool_name == "configure_experiment" and result.get("status") == "draft":
                    ChatMessage.objects.create(
                        session=self.session,
                        role='experiment_card',
                        content=json.dumps(result.get("config", {})),
                        metadata={"type": "experiment_draft", "config": result.get("config", {})},
                    )
                elif tool_name == "launch_experiment" and result.get("status") == "launched":
                    ChatMessage.objects.create(
                        session=self.session,
                        role='experiment_card',
                        content=json.dumps(result),
                        metadata={"type": "experiment_launched", "run_id": result.get("run_id")},
                    )

                tool_results.append(
                    ToolMessage(content=result_str, tool_call_id=tool_call["id"])
                )
            else:
                tool_results.append(
                    ToolMessage(content=f"Error: tool {tool_name} not found", tool_call_id=tool_call["id"])
                )

        return {"messages": tool_results}

    def _should_continue(self, state: AgentState) -> str:
        """Condition: do we have tool calls to execute?"""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"

    def run(self, user_text: str) -> dict:
        """Process a user message and return the final state."""
        # Save user message to DB
        ChatMessage.objects.create(session=self.session, role='user', content=user_text)

        # Create initial state
        config = {"configurable": {"thread_id": self.session_id}}

        # Invoke the graph
        final_state = self.graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )

        # Get the final AI response
        last_ai = None
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                last_ai = msg.content
                break

        return {
            "text": last_ai or "",
            "messages": [
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "metadata": m.metadata,
                    "created_at": m.created_at.isoformat(),
                }
                for m in self.session.messages.order_by('created_at')
            ],
        }