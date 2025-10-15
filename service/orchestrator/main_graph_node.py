from typing import Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from states.graph_states import (
    InputState,
    OverallState,
    ContextSchema,
    QueryResult,
)
from .subgraph_nodes import RetrievalSubGraph
from agents.input_agent import InputAgent
from agents.chat_agent import ChatAgent
from agents.query_agent import QueryAgent
from agents.response_agent import ResponseAgent
from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from utils.logger import get_logger

logger = get_logger(__name__)


class MainGraph:

    def __init__(self):
        self.chat_agent = ChatAgent(model_name="openai/gpt-oss-120b", temperature=0)
        self.input_agent = InputAgent(model_name="gpt-5-nano", temperature=0)
        self.query_agent = QueryAgent(model_name="gpt-4o-mini", temperature=0)
        self.response_agent = ResponseAgent(model="gpt-4o-mini", temperature=0)

        self.retrieval_subgraph = RetrievalSubGraph().subgraph
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()

    def _call_retrieval_subgraph(
        self, state: OverallState, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> Dict[str, Any]:
        print(f"In call_retrieval_subgraph with thread_id: {config['configurable']['thread_id']}")

        sub_results = []
        for subquery in state.formatted_query:
            print(f"Processing subquery: {subquery}")
            response = self.retrieval_subgraph.invoke({"subquery": subquery})

            query_result = QueryResult(
                subquery=subquery,
                search_result=response.get("search_result"),
                memories=response.get("memories"),
            )
            sub_results.append(query_result)

        return {"sub_results": sub_results}

    def _make_response(
        self, state: OverallState, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> Dict[str, str]:
        print(f"In make_response with thread_id: {config['configurable']['thread_id']}")

        seen_ids = set()
        distinct_search_results = []
        distinct_memories = []

        for sub_result in state.sub_results:
            if sub_result.search_result:
                for chunk in sub_result.search_result:
                    if chunk.get("id") not in seen_ids:
                        seen_ids.add(chunk["id"])
                        distinct_search_results.append(chunk)

            if sub_result.memories:
                distinct_memories.extend(sub_result.memories)

        print(f"No. of distinct chunks: {len(distinct_search_results)}")
        # print(f"Distinct chunks: {distinct_search_results}")
        # print(f"Distinct memories: {distinct_memories}")

        # Pass chat history to response agent for context-aware response generation
        chat_history = state.messages if state.messages else []
        response = self.response_agent.answer(
            query=state.query,
            sub_queries=state.formatted_query,
            memory=distinct_memories,
            search_result=[result["content"] for result in distinct_search_results],
            chat_history=chat_history,
        )
        print(type(response))
        print(f"Response: {response}")
        return {"final_result": response.content}

    def _input_guardrails(
        self, state: InputState, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> OverallState:
        writer = get_stream_writer()
        response = self.input_agent.run(state.query)
        is_safe = response.classification == "safe"
        writer(f"Input guardrails check: {is_safe}")
        logger.info(f"Input guardrails check: {is_safe}")
        result = {"messages": state.query, "input_guardrails": is_safe, "use_rag": False}
        if not is_safe:
            result["final_result"] = "Sorry, I can't assist with that."
        return result

    def _chat_router(
        self, state: OverallState, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> OverallState:
        chat_history = state.messages if state.messages else []

        # Get username from context if available
        logger.info(f"Runtime context from chat router: {runtime.context}")
        username = runtime.context.user_id
        logger.info(f"Username from chat router: {username}")
        response = self.chat_agent.run(
            query=state.query, chat_history=chat_history, username=username
        )
        logger.info(f"Chat router response: {response.message}")
        ai_message = AIMessage(content=response.message)
        result = {"messages": ai_message, "use_rag": response.use_rag}
        if not response.use_rag:
            result["final_result"] = response.message
        return result

    def _route_after_guardrails(self, state: OverallState, runtime: Runtime[ContextSchema]):
        return state.input_guardrails

    def _route_after_rag_usage(self, state: OverallState, runtime: Runtime[ContextSchema]):
        return state.use_rag

    def _query_formatter(
        self, state: OverallState, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> OverallState:
        writer = get_stream_writer()
        # Pass chat history to query formatter for context-aware query reformatting
        chat_history = state.messages if state.messages else []
        formatted_queries = self.query_agent.run(
            query=state.query, chat_history=chat_history
        ).queries
        writer(f"Formatted queries: {formatted_queries}")
        ai_message = AIMessage(content=f"Formatted queries: {formatted_queries}")
        return {"formatted_query": formatted_queries, "messages": ai_message}

    def _build_graph(self):
        graph_builder = StateGraph(OverallState)

        graph_builder.add_node("input_guardrails", self._input_guardrails)
        graph_builder.add_node("chat_router", self._chat_router)

        graph_builder.add_node("retrieval_subgraph", self._call_retrieval_subgraph)
        graph_builder.add_node("make_response", self._make_response)
        graph_builder.add_node("query_formatter", self._query_formatter)

        graph_builder.add_edge(START, "input_guardrails")
        graph_builder.add_conditional_edges(
            "input_guardrails", self._route_after_guardrails, {True: "chat_router", False: END}
        )
        graph_builder.add_conditional_edges(
            "chat_router", self._route_after_rag_usage, {True: "query_formatter", False: END}
        )
        graph_builder.add_edge("query_formatter", "retrieval_subgraph")
        graph_builder.add_edge("retrieval_subgraph", "make_response")
        graph_builder.add_edge("make_response", END)

        return graph_builder.compile(checkpointer=self.checkpointer)
