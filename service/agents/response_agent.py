import os
import logfire
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from process.utils.logger import get_logger
from dotenv import load_dotenv

os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"
load_dotenv()

logger = get_logger(__name__)

logfire.configure(
    token="pylf_v1_us_pLcgB8kDHBvSY5mQyqcgB1k6TdkDlw4fV3HvhPkPl98b",
    service_name="rag-response-agent",
    scrubbing=False,
)


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
            You reconstruct answers based on user queries and provided document responses.
            """
        )
        logger.info("System prompt set")

    def set_user_prompt(self):
        self.user_prompt = HumanMessagePromptTemplate.from_template(
            """
            You reconstruct answers based on
            - user orignal query
            - subqueries
            - relevant memory (if exists)
            - provided document responses

            The subqueries are the broken down version of the user query.
            In some cases the subqueries would be identical to the user query.

            Query: {query}
            SubQueries: {sub_queries}
            Related Memory : {memory}
            Search Results: {search_result}


            Give a clear and complete answer.
            Also reference the index of the document in the search results.

            If you do not have enough information to answer the query, say so.
            """
        )
        logger.info(f"User prompt set: \n {self.user_prompt}")

    def reconstruct_prompt(self):
        self.set_system_prompt()
        self.set_user_prompt()
        self.reconstruction_prompt = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.user_prompt]
        )
        logger.info("Prompt template set")

    def create_prompt(self, query, sub_queries, memory, search_result):
        return self.reconstruction_prompt.format_messages(
            query=query, sub_queries=sub_queries, memory=memory, search_result=search_result
        )

    def answer(self, query, sub_queries, memory, search_result):
        prompt = self.create_prompt(query, sub_queries, memory, search_result)
        return self.agent.invoke(prompt)
