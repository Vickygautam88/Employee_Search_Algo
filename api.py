from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import threading

# -----------------------------
# APP SETUP
# -----------------------------
app = FastAPI(
    title="Employee Search API",
    description="Semantic Employee search using FAISS",
    version="1.0"
)

FAISS_INDEX_PATH = "faiss_user.index"
JOB_DATA_PATH = "user_data.pkl"
lock = threading.Lock()

# -----------------------------
# LOAD MODEL
# -----------------------------
print("📌 Loading embedding model...")
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
print("✅ Model loaded!")

# -----------------------------
# RELOAD DATA FUNCTION
# -----------------------------
def reload_data():
    global index, df

    with lock:
        print("🔁 Reloading FAISS index + user data...")

        # Load FAISS index
        index = faiss.read_index(FAISS_INDEX_PATH)

        # Load dataframe
        with open(JOB_DATA_PATH, "rb") as f:
            df = pickle.load(f)

        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        print(f"✅ Reload complete! Total records: {len(df)}")

# Initial load
reload_data()

# -----------------------------
# Scheduler (every 5 minutes)
# -----------------------------
scheduler = BackgroundScheduler()
scheduler.add_job(reload_data, "interval", minutes=5)
scheduler.start()

# -----------------------------
# REQUEST MODEL
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    location: Optional[str] = None  # Optional location filter

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search_user(request: SearchRequest):

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    top_k = request.top_k
    search_pool = 300  # Larger pool for filtering

    # Create embedding
    query_emb = model.encode([request.query])
    query_emb = np.array(query_emb).astype("float32")

    with lock:
        distances, indices = index.search(query_emb, search_pool)
        scores = 1 / (1 + distances[0])

        results = df.iloc[indices[0]].copy()
        results["similarity_score"] = scores

    # Apply location filter if provided
    if request.location:
        results = results[
            results["user_city"].str.lower() == request.location.lower()
        ]

    # Remove weak matches
    results = results[results["similarity_score"] > 0.023]

    # Sort and limit to top_k
    results = results.sort_values(
        by="similarity_score", ascending=False
    ).head(top_k)

    return results.to_dict(orient="records")

# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.post("/search_employee")
def search_user_endpoint(request: SearchRequest):
    results = search_user(request)
    return {
        "count": len(results),
        "results": results
    }

@app.get("/")
def home():
    return {"message": "Welcome to the Employee Search API"}
