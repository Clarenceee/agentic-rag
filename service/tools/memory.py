import os
from mem0 import Memory
from process.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv(".env", override=True)

logger = get_logger(__name__)


class Mem0Memory:
    def __init__(self):
        self.config = {
            "vector_store": {"provider": "qdrant", "config": {"host": "localhost", "port": 6333}},
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
            # "embed_config": {
            #     "provider": "openai",
            #     "model": "text-embedding-3-small",
            #     "api_key": os.getenv("OPENAI_API_KEY")
            # },
            "embed_config": {"provider": "huggingface", "model": "Qwen/Qwen3-Embedding-0.6B"},
        }

        self.memory = Memory.from_config(self.config)
        logger.info("Mem0 Memory initialized")

    def add_memory(self, message: str, user_id: str) -> None:
        self.memory.add(message, user_id=user_id)
        logger.info("Memory added successfully")

    def search_memory(self, query: str, user_id: str):
        related_memories = self.memory.search(query, user_id=user_id, threshold=0.6)
        logger.info("Memory search successful")
        return related_memories
