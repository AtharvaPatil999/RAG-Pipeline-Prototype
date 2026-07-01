# 📄 RAG Pipeline using LangChain, FAISS & FastAPI

A Retrieval-Augmented Generation (RAG) application built using Python, LangChain, FAISS, HuggingFace sentence-transformers, and FastAPI.

The application allows users to upload PDF documents, converts them into embeddings, stores them in a FAISS vector database, and answers natural language questions using retrieved document context and a local language model.

---

## Features

- Upload PDF documents
- Automatic text extraction using PyPDF
- Recursive text chunking
- Embedding generation using HuggingFace Sentence Transformers
- FAISS vector database
- Semantic similarity search
- Question Answering using FLAN-T5
- REST API built with FastAPI
- Fully local (No OpenAI API Key Required)

---

## Tech Stack

- Python
- FastAPI
- LangChain
- FAISS
- HuggingFace Transformers
- Sentence Transformers
- PyPDF
- Torch

---

## Project Workflow

PDF Upload

↓

Extract Text

↓

Chunk Text

↓

Generate Embeddings

↓

Store in FAISS

↓

User asks Question

↓

Retrieve Relevant Chunks

↓

Generate Context-Aware Answer

↓

Return Answer

---

## API Endpoints

### Upload PDF

POST

```
/upload
```

Uploads a PDF and creates embeddings.

---

### Ask Question

POST

```
/ask
```

Example

```json
{
    "question":"What is Retrieval Augmented Generation?"
}
```

---

## Installation

```bash
git clone <repo>

cd rag-pipeline

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

uvicorn app:app --reload
```

---

## Future Improvements

- Multiple PDF support
- Metadata filtering
- Hybrid Search
- Streaming responses
- Local Llama 3 integration
- Agentic RAG
- Docker deployment

---
