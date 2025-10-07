import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VectorStoreConfig(BaseModel):
    """Configuration for the vector store."""

    collection_name: str = "documents"
    vector_size: int = 384  # Default for BAAI/bge-small-en-v1.5
    distance: str = "Cosine"
    on_disk: bool = True
    url: str = Field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model."""

    model_name: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    device: str = "cpu"
    normalize_embeddings: bool = True


class RetrieverConfig(BaseModel):
    """Configuration for the retriever."""

    top_k: int = 5
    use_hybrid: bool = True
    vector_weight: float = 0.7
    keyword_weight: float = 0.3


class AgentConfig(BaseModel):
    """Configuration for agents."""

    retriever_agent: Dict[str, Any] = Field(
        default_factory=lambda: {
            "name": "retriever_agent",
            "description": "An agent that retrieves relevant documents based on a query.",
            "top_k": 5,
            "use_hybrid": True,
        }
    )
    # Add more agent configurations as needed


class LLMConfig(BaseModel):
    """Configuration for the language model."""

    model_name: str = "gpt-4-turbo"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    @field_validator("api_key")
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        return v


class AppConfig(BaseModel):
    """Main application configuration."""

    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    reload: bool = os.getenv("RELOAD", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "info")


class Config(BaseModel):
    """Root configuration model."""

    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration."""
    return config


def update_config(new_config: Dict[str, Any]) -> None:
    """Update the global configuration.

    Args:
        new_config: Dictionary with configuration overrides
    """
    global config
    config = config.model_copy(update=new_config)
