# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import answer_question  # نستورد دالتك

app = FastAPI(
    title="Saudi Legal RAG API",
    description="API for asking legal questions based on Saudi regulations.",
    version="1.0.0",
)

# للسماح للفرونت-إند بالاتصال (يمكن تضييقها لاحقًا على دومين معيّن)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في الإنتاج خله دومين محدد
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    articles: list

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    result = answer_question(req.question)
    return {
        "answer": result["answer"],
        "articles": result["articles"],
    }
