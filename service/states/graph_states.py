import os
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import List, Optional, Annotated
from langgraph.graph import add_messages
from dotenv import load_dotenv

load_dotenv("/Users/clarencechan/Documents/agentic-rag/service/.env", override=True)


@dataclass
class ContextSchema:
    chat_agent_model_name: str = os.environ.get("CHAT_AGENT_MODEL", "gpt-5-nano")
    user_id: str = None

    def __post_init__(self):
        if self.user_id is None:
            self.user_id = os.environ.get("USER_ID", "default_user")


class SummarizeResponse(BaseModel):
    summary: str


class InputState(BaseModel):
    query: str


class OutputState(BaseModel):
    messages: Annotated[list, add_messages]
    input_guardrails: bool
    use_rag: bool


class FormatterState(BaseModel):
    formatted_query: Optional[List[str]] = None


class EmbeddingState(BaseModel):
    embedding: Optional[List[float]] = None


class QueryResult(BaseModel):
    subquery: str
    search_result: Optional[List[dict]] = None
    memories: Optional[List[dict]] = None
    rerank_score: Optional[float] = None


class OverallState(BaseModel):
    query: str
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    formatted_query: Optional[List[str]] = None
    input_guardrails: bool = False
    use_rag: bool = False
    use_web: bool = False
    sub_results: List[QueryResult] = Field(default_factory=list)
    final_result: str = ""
    chat_summary: str = ""
    approved: Optional[bool] = None
