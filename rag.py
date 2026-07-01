import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from transformers import pipeline


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_PATH = "vectorstore"


embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)


def process_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    db = FAISS.from_documents(
        chunks,
        embedding
    )

    db.save_local(VECTOR_PATH)

    return len(chunks)


def answer_question(question):

    db = FAISS.load_local(
        VECTOR_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Context:

{context}

Question:

{question}

Answer based only on the context.
"""

    response = generator(
        prompt,
        max_new_tokens=200
    )

    return {
        "answer": response[0]["generated_text"],
        "sources": [d.page_content[:200] for d in docs]
    }