from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import json

from utils import (
    clean_question,
    calculate_similarity,
    get_random_vector,
    filter_high_confidence,
    log_query
)
from data import get_all_documents, get_document_by_id

# Create FastAPI app
app = FastAPI(
    title="Mini RAG Simulator",
    description="Production RAG simulation using Python fundamentals"
)

# Request model
class Question(BaseModel):
    question: str
    threshold: float = 0.5
    num_results: int = 3

# Home endpoint
@app.get("/")
def home():
    return {
        "name": "Mini RAG Simulator",
        "version": "1.0",
        "endpoints": ["/ask", "/documents", "/logs", "/health"]
    }

# Health check
@app.get("/health")
def health():
    docs = get_all_documents()
    return {
        "status": "healthy",
        "total_documents": len(docs)
    }

# Get all documents
@app.get("/documents")
def get_documents():
    docs = get_all_documents()
    return {
        "total": len(docs),
        "documents": [doc.to_dict() for doc in docs]
    }

# Get one document
@app.get("/documents/{doc_id}")
def get_document(doc_id: int):
    doc = get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document {doc_id} not found"
        )
    return doc.to_dict()

# Main RAG endpoint
@app.post("/ask")
def ask(body: Question):
    try:
        # Step 1 — clean question
        clean_q = clean_question(body.question)

        # Step 2 — create query vector
        query_vector = get_random_vector()

        # Step 3 — get all documents
        documents = get_all_documents()

        # Step 4 — calculate similarity scores
        results = []
        for doc in documents:
            score = calculate_similarity(
                query_vector,
                doc.vector
            )
            results.append({
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "source": doc.source,
                "score": score
            })

        # Step 5 — sort by score
        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        # Step 6 — filter high confidence
        high_confidence = filter_high_confidence(
            results,
            body.threshold
        )

        # Step 7 — get top results
        top_results = high_confidence[:body.num_results]

        # Step 8 — log query
        log = log_query(clean_q, top_results)

        # Step 9 — return response
        return {
            "question": clean_q,
            "total_results": len(top_results),
            "results": top_results,
            "avg_score": log["avg_score"],
            "timestamp": log["timestamp"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )

# View logs
@app.get("/logs")
def get_logs():
    try:
        logs = []
        with open("query_log.json", "r") as f:
            for line in f:
                logs.append(json.loads(line))
        return {
            "total_queries": len(logs),
            "logs": logs
        }
    except FileNotFoundError:
        return {
            "total_queries": 0,
            "logs": []
        }
