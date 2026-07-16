# Mini RAG Simulator

A production-style RAG (Retrieval Augmented Generation) simulator
built with Python fundamentals — FastAPI, NumPy, Pandas, and OOP.

## What This Project Does
- Takes user questions via REST API
- Cleans and processes questions
- Calculates document similarity using NumPy vectors
- Filters high confidence results
- Logs all queries with timestamps
- Returns structured JSON responses

## Tech Stack
- FastAPI — REST API framework
- NumPy — vector similarity calculations
- Pandas — data handling
- Pydantic — request validation
- Uvicorn — ASGI server
- Python-dotenv — environment variables

## Project Structure
- main.py — FastAPI app and RAG logic
- utils.py — helper functions
- data.py — document database
- requirements.txt — dependencies

## API Endpoints
- GET / — home
- GET /health — health check
- GET /documents — list all documents
- GET /documents/{id} — get one document
- POST /ask — ask a question
- GET /logs — view query history

## How to Run
pip install -r requirements.txt
uvicorn main:app --reload

## Author
Rishi — M.Tech AI & Data Science
Building toward Production LLM/GenAI Engineer role
