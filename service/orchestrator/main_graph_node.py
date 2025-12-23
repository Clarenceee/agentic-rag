from typing import Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig
from states.graph_states import (
    InputState,
    OverallState,
    ContextSchema,
    QueryResult,
    SummarizeResponse,
)
from .subgraph_nodes import RetrievalSubGraph
from agents.input_agent import InputAgent
from agents.chat_agent import ChatAgent
from agents.query_agent import QueryAgent
from agents.response_agent import ResponseAgent
from tools.reranker import Reranker
from tools.web_search import web_search
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.config import get_stream_writer
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from langfuse import get_client
from utils.logger import get_logger

langfuse = get_client()
logger = get_logger(__name__)


class MainGraph:

    def __init__(self):
        self.chat_agent = ChatAgent(model_name="gpt-4o-mini", temperature=0)
        self.input_agent = InputAgent(model_name="gpt-5-nano", temperature=0)
        self.query_agent = QueryAgent(model_name="gpt-4o-mini", temperature=0)
        self.response_agent = ResponseAgent(model="gpt-4o-mini", temperature=0)
        self.summarize_model = ChatOpenAI(model_name="gpt-5-mini", temperature=0)
        self.summarize_model = self.summarize_model.with_structured_output(
            schema=SummarizeResponse,
        )

        self.retrieval_subgraph = RetrievalSubGraph().subgraph
        self.reranker = Reranker()

        # Initialize checkpointer
        DB_URI = "postgresql://clarencechan@172.17.0.1:5432/postgres?sslmode=disable"

        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }

        conn = Connection.connect(DB_URI, **connection_kwargs)
        self.checkpointer = PostgresSaver(conn)
        self.checkpointer.setup()
        logger.info("Postgres checkpointer initialized")

        # Verify langfuse
        if langfuse.auth_check():
            logger.info("Langfuse client is authenticated and ready!")
        else:
            logger.info("Authentication failed. Please check your credentials and host.")

        self.graph = self._build_graph()

    def _input_guardrails(
        self, state: InputState, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> OverallState:
        writer = get_stream_writer()
        response = self.input_agent.run(state.query)
        is_safe = response.classification == "safe"
        logger.info(f"Input guardrails state: {state}")
        writer(f"Input guardrails check: {is_safe}")
        logger.info(f"Input guardrails check: {is_safe}")
        result = {"input_guardrails": is_safe, "use_rag": False}
        if not is_safe:
            result["messages"] = [
                HumanMessage(content=state.query),
                AIMessage(content="Sorry, I can't assist with that."),
            ]
        return result

    def _route_after_guardrails(self, state: OverallState, runtime: Runtime[ContextSchema]):
        """Function to determine node based on input guardrails result (continue / END)"""
        return state.input_guardrails

    def _chat_router(
        self, state: OverallState, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> OverallState:
        chat_history = state.messages if state.messages else []
        logger.info(f"Chat router input state: {state}")
        # Get username from context if available
        logger.info(f"Runtime context from chat router: {runtime.context}")
        username = runtime.context.user_id
        logger.info(f"Username from chat router: {username}")
        response = self.chat_agent.run(
            query=state.query, chat_history=chat_history, username=username
        )
        logger.info(f"Chat router response: {response.message}")
        human_message = HumanMessage(content=state.query)
        ai_message = AIMessage(content=response.message)
        result = {
            "messages": (
                [human_message]
                if response.use_rag or response.use_web
                else [human_message, ai_message]
            ),
            "use_rag": response.use_rag,
            "use_web": response.use_web,
            "sub_results": [],
        }
        if not response.use_rag and not response.use_web:
            result["final_result"] = response.message
        return result

    def _route_after_rag_usage(self, state: OverallState, runtime: Runtime[ContextSchema]):
        """Function to determine node based on RAG usage result (continue / END)"""
        if state.use_rag:
            return "use_rag"
        elif state.use_web:
            return "use_web"
        else:
            return END

    def _call_retrieval_subgraph(
        self, state: OverallState, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> Dict[str, Any]:

        sub_results = []
        for subquery in state.formatted_query:
            logger.info(f"Processing subquery: {subquery}")
            response = self.retrieval_subgraph.invoke({"subquery": subquery})

            query_result = QueryResult(
                subquery=subquery,
                search_result=response.get("search_result"),
                memories=response.get("memories"),
            )
            sub_results.append(query_result)

        # Apply reranking to retreived documents
        documents = []
        chunk_positions = []
        seen_ids = set()

        for result_idx, result in enumerate(sub_results):
            if result.search_result:
                for chunk_idx, chunk in enumerate(result.search_result):
                    if chunk.get("id") not in seen_ids:
                        seen_ids.add(chunk["id"])
                        documents.append(chunk["content"])
                        chunk_positions.append((result_idx, chunk_idx))

        logger.info(f"Calling reranker with: {state.query} and {len(documents)} documents")
        rerank_scores = self.reranker.run(queries=[state.query], documents=documents)
        logger.info(f"Reranker scores : {rerank_scores}")

        for (result_idx, chunk_idx), score in zip(chunk_positions, rerank_scores):
            sub_results[result_idx].search_result[chunk_idx]["rerank_score"] = score

        return {"sub_results": sub_results}

    def _make_response(
        self, state: OverallState, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> Dict[str, str]:

        seen_ids = set()
        distinct_search_results = []
        distinct_memories = []

        for sub_result in state.sub_results:
            if sub_result.search_result:
                for chunk in sub_result.search_result:
                    if chunk.get("id") not in seen_ids and chunk.get("rerank_score", 0) > 0.4:
                        seen_ids.add(chunk["id"])
                        distinct_search_results.append(chunk)

            if sub_result.memories:
                distinct_memories.extend(sub_result.memories)

        logger.info(f"No. of distinct chunks: {len(distinct_search_results)}")

        # Pass chat history to response agent for context-aware response generation
        chat_history = state.messages if state.messages else []
        response = self.response_agent.answer(
            query=state.query,
            sub_queries=state.formatted_query,
            memory=distinct_memories,
            search_result=[result["content"] for result in distinct_search_results],
            chat_history=chat_history,
        )
        logger.info(f"Response: {response}")
        ai_message = AIMessage(content=response.content)
        return {"messages": [ai_message], "final_result": response.content}

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
        return {"formatted_query": formatted_queries}

    def _should_summarize(self, state: OverallState) -> str:
        messages = state.messages
        print(f"Current message count: {len(messages)}")
        if len(messages) >= 10:
            print(f"Current message count: {len(messages)} => Calling Summarizer")
            return True
        return False

    def _chat_summarizer(self, state: OverallState) -> OverallState:
        message_count = len(state.messages)
        prev_summarization = state.chat_summary
        print(f"In summarizing node at {message_count} messages")
        if prev_summarization:
            summary_message = (
                f"This is a summary of the conversation to date: {prev_summarization}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        model_input = state.messages + [HumanMessage(content=summary_message)]
        response = self.summarize_model.invoke(model_input)

        to_delete = [RemoveMessage(id=m.id) for m in state.messages]
        new_messages = to_delete + [AIMessage(content=response.summary)]
        return {"messages": new_messages, "chat_summary": response.summary}

    def _approval_node(self, state: OverallState):
        approved = interrupt(f"Do you approve this web search for '{state.query}' ?")
        return {"approved": approved}

    def _approval_routing(self, state: OverallState):
        return state.approved

    def _call_web_search_node(self, state: OverallState):
        web_search_response = web_search(state.query)
        return {"messages": [AIMessage(content=web_search_response.content[0]["text"])]}

    def _build_graph(self):
        graph_builder = StateGraph(OverallState)

        graph_builder.add_node("input_guardrails", self._input_guardrails)
        graph_builder.add_node("chat_summarizer", self._chat_summarizer)
        graph_builder.add_node("chat_router", self._chat_router)

        graph_builder.add_node("retrieval_subgraph", self._call_retrieval_subgraph)
        graph_builder.add_node("make_response", self._make_response)
        graph_builder.add_node("query_formatter", self._query_formatter)

        graph_builder.add_node("approval_node", self._approval_node)
        graph_builder.add_node("call_web_search_node", self._call_web_search_node)

        graph_builder.add_conditional_edges(
            START, self._should_summarize, {True: "chat_summarizer", False: "input_guardrails"}
        )
        graph_builder.add_edge("chat_summarizer", "input_guardrails")
        graph_builder.add_conditional_edges(
            "input_guardrails", self._route_after_guardrails, {True: "chat_router", False: END}
        )
        graph_builder.add_conditional_edges(
            "chat_router",
            self._route_after_rag_usage,
            {"use_rag": "query_formatter", "use_web": "approval_node", END: END},
        )
        graph_builder.add_conditional_edges(
            "approval_node", self._approval_routing, {True: "call_web_search_node", False: END}
        )
        graph_builder.add_edge("call_web_search_node", END)
        graph_builder.add_edge("query_formatter", "retrieval_subgraph")
        graph_builder.add_edge("retrieval_subgraph", "make_response")
        graph_builder.add_edge("make_response", END)

        return graph_builder.compile(checkpointer=self.checkpointer)
