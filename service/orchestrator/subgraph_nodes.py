import os
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# from langgraph.checkpoint.memory import InMemorySaver
from tools.embedding_generator import EmbeddingGenerator
from tools.vector_store import QdrantVectorStore
from tools.memory import Mem0Memory
from states.graph_states import EmbeddingState, QueryResult, ContextSchema
from langgraph.config import get_stream_writer
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from langfuse import get_client
from utils.logger import get_logger

logger = get_logger(__name__)
langfuse = get_client()


class RetrievalSubGraph:
    """
    This retireval subgraph includes embedding generation,
    vector search, and memory search functionality.
    """

    def __init__(self):
        self.collection_name = os.getenv("QDRANT_COLLECTION", "nba_rules_test")
        self.qdrant_url = os.getenv("QDRANT_URL")

        # Initialize components
        self.emb_generator = EmbeddingGenerator()
        self.mem_zero = Mem0Memory()
        self.qdrant_store = QdrantVectorStore(
            collection_name=self.collection_name,
            vector_size=self.emb_generator.embedding_dimension,
            qdrant_url=self.qdrant_url,
        )

        # Initialize checkpointer
        DB_URI = "postgresql://clarencechan@localhost:5432/postgres?sslmode=disable"

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

        # Build the subgraph
        self.subgraph = self._build_subgraph()

    @staticmethod
    def _normalize_scored_points(points):
        return [
            {
                "id": p.id,
                "score": p.score,
                "source": p.payload.get("source"),
                "page": p.payload.get("page"),
                "chunk_index": p.payload.get("chunk_index"),
                "content": p.payload.get("content"),
            }
            for p in points
        ]

    def generate_embedding(
        self, state: QueryResult, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> EmbeddingState:
        writer = get_stream_writer()
        # print("In generate_embedding node with thread_id: ",
        #       config["configurable"]["thread_id"],
        #       " with user = ", runtime.context.user_id)
        writer(f"Generating embedding for {state.subquery}")

        embedding = self.emb_generator.generate_embedding(state.subquery)
        # print(f"Generated embedding shape: {embedding.shape}")
        return {"embedding": embedding}

    def vector_search(
        self, state: EmbeddingState, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> QueryResult:
        """Perform vector search using the generated embedding."""
        writer = get_stream_writer()
        # print("In vector_search node with thread_id: ",
        #       config["configurable"]["thread_id"],
        #       " with user = ", runtime.context.user_id)
        writer("Performing similarity search with database")
        results = self.qdrant_store.search(state.embedding)
        normalized_results = self._normalize_scored_points(results)
        return {"search_result": normalized_results}

    def memory_search(
        self, state: QueryResult, config: RunnableConfig, runtime: Runtime[ContextSchema]
    ) -> QueryResult:
        """Search for relevant memories."""
        writer = get_stream_writer()
        # print("In memory_search node with thread_id: ",
        #       config["configurable"]["thread_id"],
        #       " with user = ", runtime.context.user_id)
        writer("Performing memory search for any relevant preferences")
        memories = self.mem_zero.search_memory(state.subquery, user_id=runtime.context.user_id)
        return {"memories": memories.get("results", [])}

    def _build_subgraph(self):
        subgraph_builder = StateGraph(QueryResult, context_schema=ContextSchema)

        # Add nodes
        subgraph_builder.add_node("generate_embedding", self.generate_embedding)
        subgraph_builder.add_node("vector_search", self.vector_search)
        subgraph_builder.add_node("memory_search", self.memory_search)

        # Define edges
        subgraph_builder.add_edge(START, "generate_embedding")
        subgraph_builder.add_edge(START, "memory_search")
        subgraph_builder.add_edge("generate_embedding", "vector_search")
        subgraph_builder.add_edge("vector_search", END)
        subgraph_builder.add_edge("memory_search", END)

        logger.info("Compiling subgraph")
        return subgraph_builder.compile(checkpointer=self.checkpointer)
