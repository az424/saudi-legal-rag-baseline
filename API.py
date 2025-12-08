# api.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import answer_question
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime
import os
from dotenv import load_dotenv

# =====================================
# 1) Load environment variables
# =====================================
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# =====================================
# 2) Database setup with SQLAlchemy
# =====================================
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =====================================
# 3) Evaluation table definition
# =====================================
class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text)
    answer = Column(Text)
    accuracy = Column(Integer)
    clarity = Column(Integer)
    legal_depth = Column(Integer)
    relevance = Column(Integer)
    overall_score = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create the table if it does not exist
Base.metadata.create_all(bind=engine)

# =====================================
# 4) FastAPI initialization
# =====================================
app = FastAPI(
    title="Saudi Legal RAG API",
    description="API for asking legal questions based on Saudi regulations.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# 5) Request/Response Models
# =====================================
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    articles: list

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    accuracy: int
    clarity: int
    legal_depth: int
    relevance: int
    overall_score: int

# =====================================
# 6) Health check
# =====================================
@app.get("/health")
def health_check():
    return {"status": "ok"}

# =====================================
# 7) Main RAG endpoint
# =====================================
@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    result = answer_question(req.question)
    return {
        "answer": result["answer"],
        "articles": result["articles"],
    }

# =====================================
# 8) NEW: Evaluation submission endpoint
# =====================================
@app.post("/evaluate")
def submit_evaluation(req: EvaluationRequest, db: Session = Depends(get_db)):
    evaluation = Evaluation(
        question=req.question,
        answer=req.answer,
        accuracy=req.accuracy,
        clarity=req.clarity,
        legal_depth=req.legal_depth,
        relevance=req.relevance,
        overall_score=req.overall_score,
    )
    db.add(evaluation)
    db.commit()
    return {"status": "saved", "evaluation_id": evaluation.id}
