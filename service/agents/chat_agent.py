import os
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel
from langchain.prompts import MessagesPlaceholder
from utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)
os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"


class ChatAgentResponse(BaseModel):
    use_rag: bool
    message: str


class ChatAgent:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
        self.create_agent()
        self.reconstruct_prompt()

    def create_agent(self):
        self.model = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        # self.model = ChatOpenAI(
        #     model=self.model_name,
        #     temperature=self.temperature,
        #     api_key=os.getenv("OPENROUTER_API_KEY"),
        #     base_url="https://openrouter.ai/api/v1",
        #     max_retries=5,
        # )
        self.model = self.model.with_structured_output(
            schema=ChatAgentResponse,
        )
        logger.info(f"Chat agent created with model: {self.model}")

    def set_system_prompt(self):
        self.system_prompt = SystemMessagePromptTemplate.from_template(
            """
            You are an expert NBA rules assistant.
            For greetings or general small talk, reply briefly and naturally without using RAG.
            For any query related to NBA rules (interpretations, violations, gameplay, etc.),
            set use_rag = true and return "This is a NBA rule related question".
            """
        )

    def set_user_prompt(self):
        self.user_prompt = HumanMessagePromptTemplate.from_template(
            """
            User Name: {username}
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
        self.chain = self.prompt_template | self.model

    def run(self, query, chat_history, username=None):
        try:
            # Get the formatted prompt before invoking the chain
            logger.info(f"Chat agent input history: {chat_history}")
            formatted_messages = self.prompt_template.format_messages(
                query=query, chat_history=chat_history, username=username
            )

            logger.info(f"Formatted chat agent prompt: {formatted_messages}")

            # Invoke the chain with the same parameters
            response = self.chain.invoke(
                {"query": query, "chat_history": chat_history, "username": username}
            )
            logger.info(f"Type of response: {type(response)}")
            logger.info(f"Chat agent response: {response}")

            # Ensure response has the expected structure
            if not hasattr(response, "use_rag") or not hasattr(response, "message"):
                logger.warning(f"Unexpected response format: {response}")
                return ChatAgentResponse(use_rag=False, message=str(response))
            return response
        except Exception as e:
            logger.error(f"Error in ChatAgent: {str(e)}", exc_info=True)
            return ChatAgentResponse(
                use_rag=False,
                message="I encountered an error processing your request. Please try again.",
            )
