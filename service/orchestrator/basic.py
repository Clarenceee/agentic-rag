import threading
import asyncio
from typing import List
from process.embedding_generator import EmbeddingGenerator
from process.memory import Mem0Memory
from process.vector_store import QdrantVectorStore
from process.retriever_agent import RetrieverAgent
from process.query_agent import QueryAgent
from process.input_agent import InputAgent
from process.chat_agent import ChatAgent
from dotenv import load_dotenv

load_dotenv("/Users/clarencechan/Documents/agentic-rag/process/.env", override=True)


class NbaRag:
    # Normal orchestration of workflow with async processing
    def __init__(self, user_id):
        self.user_id = user_id
        self.collection_name = "nba_rules_test"
        self.qdrant_url = "http://localhost:6333"
        self.embGenerator = EmbeddingGenerator()
        self.qdrantStore = QdrantVectorStore(
            collection_name=self.collection_name,
            vector_size=self.embGenerator.embedding_dimension,
            qdrant_url=self.qdrant_url,
        )
        self.mem = Mem0Memory()
        self.retrieverAgent = RetrieverAgent(model="gpt-4o-mini", temperature=0)
        self.queryAgent = QueryAgent(model_name="gpt-4o-mini", temperature=0)
        self.inputAgent = InputAgent(model_name="gpt-5-nano", temperature=0)
        self.chatAgent = ChatAgent(model_name="gpt-4o-mini", temperature=0)

    async def _process_sub_query(self, formatted_query: str) -> tuple[list, list]:
        """Process a single sub-query to get search results and memories."""
        print(f"Processing sub-query: {formatted_query}")

        # Generate embedding for the query
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None,  # Use default ThreadPoolExecutor
            self.embGenerator.generate_embedding,
            formatted_query,
        )

        # Search in vector store
        search_result = await asyncio.get_event_loop().run_in_executor(
            None, self.qdrantStore.search, query_embedding
        )

        # Search in memory
        related_memories = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.mem.search_memory(formatted_query, user_id=self.user_id)
        )

        return search_result, related_memories.get("results", [])

    def _save_memory_background(self, query: str, response: str) -> None:
        """Save conversation to memory in a background thread."""
        try:
            new_messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
            self.mem.add_memory(new_messages, user_id=self.user_id)
        except Exception as e:
            print(f"Error saving memory: {str(e)}")

    async def process_queries_async(self, queries: List[str]) -> tuple[list, list]:
        """Process multiple sub-queries concurrently."""
        tasks = [self._process_sub_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_search_results = []
        all_memories = []

        for search_result, memories in results:
            if not isinstance(search_result, Exception):
                all_search_results.append(search_result)
            if memories and not isinstance(memories, Exception):
                all_memories.extend(memories)

        return all_search_results, all_memories

    def main(self, query: str) -> str:
        """Main entry point for processing queries with async sub-query processing."""
        # Run input safety check
        safety_check = self.inputAgent.run(query=query)
        print(f"Input query safety check: {safety_check.classification}")

        if safety_check.classification != "safe":
            return "I am sorry, but I cannot answer your question."

        # Get chat response to determine if we need RAG
        chat_response = self.chatAgent.run(query=query)
        print(f"Chat response: {chat_response}")

        if not chat_response.use_rag:
            # For non-RAG responses, just return the chat response
            return chat_response.message

        # Generate sub-queries for RAG
        formatted_queries = self.queryAgent.run(query=query).queries
        print(f"Formatted queries: {formatted_queries}")

        # Process sub-queries asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            all_search_results, all_memories = loop.run_until_complete(
                self.process_queries_async(formatted_queries)
            )
        finally:
            loop.close()

        # Process and deduplicate search results
        seen_ids = set()
        distinct_search_results = []

        for search_result in all_search_results:
            for chunk in search_result:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    distinct_search_results.append(chunk)

        # Generate final response
        response = self.retrieverAgent.answer(
            query=query,
            sub_queries=formatted_queries,
            memory=[all_memories] if all_memories else [],
            search_result=distinct_search_results,
        )

        # Save conversation to memory in background
        thread = threading.Thread(
            target=self._save_memory_background,
            args=(query, response if isinstance(response, str) else response.content),
        )
        thread.daemon = True
        thread.start()

        return response if isinstance(response, str) else response.content


# Normal chat failure
# Logo
# Icon
# User personlization
