from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel
from langchain.prompts import MessagesPlaceholder
from process.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


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
        # self.model = self.model.bind_tools(
        #     tools=[],
        #     response_format=ChatAgentResponse,
        #     strict=True,
        # )
        self.model = self.model.with_structured_output(
            schema=ChatAgentResponse,
            include_raw=False,
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
        logger.info("System prompt set")

    def set_user_prompt(self):
        self.user_prompt = HumanMessagePromptTemplate.from_template(
            """
            User Query: {query}
            """
        )
        logger.info("User prompt set")

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
        logger.info("Agent chain set.")

    def run(self, query, chat_history):
        return self.chain.invoke({"query": query, "chat_history": chat_history})
