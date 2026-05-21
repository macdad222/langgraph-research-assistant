from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(model: ChatOpenAI, checkpointer: AsyncRedisSaver):
    async def call_model(state: AgentState) -> AgentState:
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.set_entry_point("model")
    graph.add_edge("model", END)
    return graph.compile(checkpointer=checkpointer)


def initial_messages(prompt: str) -> list:
    return [
        SystemMessage(content="You are a concise, helpful AI agent running through LangGraph."),
        HumanMessage(content=prompt),
    ]
