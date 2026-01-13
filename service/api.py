import logging
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from orchestrator.main_graph_node import MainGraph
from states.graph_states import ContextSchema, OverallState

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the graph once
    global main_graph
    try:
        main_graph = MainGraph().graph
        logger.info("MainGraph initialized successfully during startup")
    except Exception as e:
        logger.error(f"Failed to initialize MainGraph: {str(e)}", exc_info=True)
        raise RuntimeError(f"Could not initialize MainGraph: {e}")

    yield
    logger.info("Application shutdown - cleaning up if necessary")


class QueryRequest(BaseModel):
    query: str
    user_id: str


class QueryResponse(BaseModel):
    response: str
    user_id: str
    metadata: Optional[Dict[str, Any]]


app = FastAPI(
    title="Agentic RAG API",
    description="API for interacting with the Agentic RAG system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/ping")
async def health_check():
    return {
        "status": "pong",
        "version": "0.1.0",
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        logger.info(f"Received webhook query: {request.query} from user: {request.user_id}")
        user_id = request.user_id
        context = ContextSchema(user_id=request.user_id)

        input_state = OverallState(
            query=request.query,
        )

        try:
            result = main_graph.invoke(
                input_state,
                config={"configurable": {"thread_id": user_id}},
                context=context,
            )
        except Exception as e:
            logger.error("Error executing graph", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing query: {str(e)}",
            )

        if result.get("__interrupt__", []):
            final_response = "Human Action needed for web search."
        else:
            final_response = result.get("final_result")

        return QueryResponse(
            response=final_response,
            user_id=user_id,
            metadata=result,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        )


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=5050,
        reload=True,
        log_level="info",
    )
