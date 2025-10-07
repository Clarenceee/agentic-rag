# Agentic RAG System

A sophisticated Retrieval-Augmented Generation (RAG) framework that combines the power of large language models with dynamic knowledge retrieval. The system enables complex, multi-step reasoning and information synthesis across diverse knowledge domains through its modular, agent-based architecture.

This implementation demonstrates how to build production-grade RAG applications with advanced capabilities such as:
- Multi-agent orchestration
- Dynamic workflow management
- Context-aware information retrieval
- Stateful conversation handling

## 🏗️ System Architecture

### Core Components

#### 1. Frontend Layer
- **Streamlit-based Web Interface**: Interactive chat interface for user interactions
- **Session Management**: Maintains conversation history and context

#### 2. Service Layer
- **Main Application** (`main.py`): Entry point handling user requests and orchestrating the RAG pipeline
- **Orchestrator** (`orchestrator/`): Implements workflow management with two approaches:
  - `graph.py`: Advanced orchestration using LangGraph for complex, stateful workflows with dynamic routing and multi-step reasoning
  - `basic.py`: Standard implementation using LangChain's built-in tools and agents

#### 3. Agent Layer (`agents/`)
- Specialized agents for different aspects of query processing:

#### 4. Tooling Layer (`tools/`)
- Utility functions and tools that support the RAG pipeline
- Integration with external services and APIs

#### 5. Utilities (`utils/`)
- Helper functions and common utilities

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.11+
- **AI/ML**:
  - OpenAI GPT models for text generation
  - Vector embeddings for semantic search
- **Data Storage**:
  - Vector database for efficient similarity search
  - Document store for rule storage
