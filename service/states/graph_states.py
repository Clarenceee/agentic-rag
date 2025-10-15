import os
import operator
from pydantic import BaseModel
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


class OverallState(BaseModel):
    query: str
    messages: Annotated[list, add_messages]
    formatted_query: Optional[List[str]] = None
    input_guardrails: bool = False
    use_rag: bool = False
    sub_results: Annotated[List[QueryResult], operator.add] = []
    final_result: str = ""
