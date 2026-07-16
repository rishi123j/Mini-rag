import numpy as np
import json
import datetime

# ---- String cleaning ----
def clean_question(question):
    question = question.strip()
    question = question.lower()
    question = question.replace("??", "?")
    question = question.replace("!!", "!")
    return question

# ---- Similarity scoring ----
def calculate_similarity(query_vector, doc_vector):
    return round(float(np.dot(query_vector, doc_vector)), 2)

def get_random_vector(size=5):
    return np.random.rand(size)

# ---- Filtering ----
def filter_high_confidence(results, threshold=0.5):
    return [r for r in results if r["score"] >= threshold]

def filter_low_confidence(results, threshold=0.5):
    return [r for r in results if r["score"] < threshold]

# ---- Logging ----
def log_query(question, results):
    log = {
        "timestamp": str(datetime.datetime.now()),
        "question": question,
        "num_results": len(results),
        "avg_score": round(
            sum(r["score"] for r in results) / len(results), 2
        ) if results else 0
    }
    with open("query_log.json", "a") as f:
        f.write(json.dumps(log) + "\n")
    return log
