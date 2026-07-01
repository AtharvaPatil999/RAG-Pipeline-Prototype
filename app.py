import os

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

from pydantic import BaseModel

from rag import process_pdf
from rag import answer_question


app = FastAPI()

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class Question(BaseModel):
    question: str


@app.post("/upload")

async def upload_pdf(file: UploadFile = File(...)):

    pdf_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    chunks = process_pdf(pdf_path)

    return {
        "message": "PDF processed successfully",
        "chunks": chunks
    }


@app.post("/ask")

def ask_question(data: Question):

    return answer_question(data.question)