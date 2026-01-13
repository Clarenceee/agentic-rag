from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from pydantic import BaseModel, Field
from typing import List
from utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


class SearchQueryList(BaseModel):
    """A list of simple and concise search queries."""

    queries: List[str] = Field(
        ..., description="A list of search queries derived from the user's complex question."
    )


class QueryAgent:
    """Agent responsible for query formatter."""

    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
        self.create_agent()
        self.reconstruct_prompt()

    def create_agent(self):
        self.model = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        self.agent = self.model.with_structured_output(SearchQueryList)
        logger.info(f"Query Reformatter agent created with model: {self.model_name}")

    def set_system_prompt(self):
        self.system_prompt = SystemMessagePromptTemplate.from_template(
            """
            You are an expert query generator for a NBA rule RAG system.

            Your task is to take a user's complex or ambiguous question and break it
            down into a set of multiple, simple, and concise search queries.

            The goal is to maximize the chances of finding relevant information in a knowledge base.

            Consider the chat history to understand context and references for ambiguous queries.
            Each query should be standalone and include necessary context from the conversation.
            Only return maximum of 3 subqueries.
            """
        )

    def set_user_prompt(self):
        self.user_prompt = HumanMessagePromptTemplate.from_template(
            """
            Break down the user query into a set of search queries.
            If the query is simple and clear, return it as is.

            User Query: {query}
            """
        )

    def reconstruct_prompt(self):
        self.set_system_prompt()
        self.set_user_prompt()
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                self.system_prompt,
                MessagesPlaceholder(variable_name="chat_history"),
                self.user_prompt,
            ]
        )
        self.chain = self.prompt_template | self.agent

    def run(self, query, chat_history=None):
        if chat_history is None:
            chat_history = []
        return self.chain.invoke({"query": query, "chat_history": chat_history})
