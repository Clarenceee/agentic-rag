import os
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from utils.logger import get_logger
from dotenv import load_dotenv

os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"
load_dotenv()

logger = get_logger(__name__)


class ResponseAgent:
    """Agent responsible for retrieving relevant documents."""

    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature
        self.create_answering_agent()
        self.reconstruct_prompt()

    def create_answering_agent(self):
        self.agent = ChatOpenAI(model=self.model, temperature=self.temperature)
        logger.info(f"Retriever agent created with model: {self.model}")

    def set_system_prompt(self):
        self.system_prompt = ChatPromptTemplate.from_template(
            """
            You are an expert NBA Rules Assistant designed to provide accurate and
            user-friendly answers about NBA rules and regulations.

            Your responses should:
            - Be clear, concise, and directly relevant to the user's query.
            - Focus exclusively on official NBA rules, procedures, or related regulations.
            - Use chat history, memory, search results to ensure continuity and accuracy.
            - Cite specific sources (e.g., search result index [1]) when applicable.
            - Avoid speculative answers or unrelated topics.
            - If information is missing, clearly state what is needed or admit limitations.
            - Adopt a professional yet approachable tone.

            Your goal is to help users understand NBA rules with precision and clarity.
            """
        )

    def set_user_prompt(self):
        self.user_prompt = HumanMessagePromptTemplate.from_template(
            """
            You reconstruct answers based on:
            - Chat history (for context and continuity)
            - User original query
            - Relevant memory (if exists)
            - Provided document responses

            Query: {query}
            Related Memory : {memory}
            Search Results: {search_result}
            Chat History: {chat_history}

            Give a clear and complete answer that considers the conversation context.
            Also reference the index of the document in the search results.

            If you do not have enough information to answer the query, say so.
            """
        )

    def reconstruct_prompt(self):
        self.set_system_prompt()
        self.set_user_prompt()
        self.reconstruction_prompt = ChatPromptTemplate.from_messages(
            [
                self.system_prompt,
                MessagesPlaceholder(variable_name="chat_history"),
                self.user_prompt,
            ]
        )

    def create_prompt(self, query, sub_queries, memory, search_result, chat_history=None):
        if chat_history is None:
            chat_history = []
        return self.reconstruction_prompt.format_messages(
            query=query,
            sub_queries=sub_queries,
            memory=memory,
            search_result=search_result,
            chat_history=chat_history,
        )

    def answer(self, query, sub_queries, memory, search_result, chat_history=None):
        if chat_history is None:
            chat_history = []
        prompt = self.create_prompt(query, sub_queries, memory, search_result, chat_history)
        return self.agent.invoke(prompt)
