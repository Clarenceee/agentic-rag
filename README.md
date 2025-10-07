# Agentic RAG System

Advanced Retrieval-Augmented Generation system with agentic capabilities for sophisticated document querying and reasoning.

## 🚀 Features

- **Hybrid Retrieval**: Combines vector and keyword search
- **Agentic Workflows**: Multi-step reasoning with LangGraph
- **Structured Output**: Pydantic-validated responses
- **Extensible**: Modular design for easy customization

## 🛠️ Tech Stack

- Python 3.11+
- OpenAI GPT models
- Qdrant vector store
- FastAPI backend
- LangChain & LangGraph

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the API**:
   ```bash
   uvicorn main:app --reload
   ```

4. **Access docs**: http://localhost:8000/docs

## 📁 Project Structure

```
agentic-rag/
├── data/           # Source documents
├── ingestion/      # Document processing
├── retriever/      # Search components
├── agents/         # Agent definitions
├── workflows/      # LangGraph workflows
├── models/         # Data models
├── config.py       # App config
└── main.py         # FastAPI app
```

## 📚 Usage Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    rag = get_rag_system()
    result = await rag.query(request.query)
    return {"response": result["response"], "documents": result["documents"]}
```

## 📝 Requirements

- Python 3.11+
- Qdrant (local or cloud)
- OpenAI API key

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and open a PR

## 📄 License

MIT# agentic-rag
