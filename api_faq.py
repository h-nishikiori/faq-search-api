import os
import numpy as np
import json
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

app = FastAPI()

# 💡 超軽量モデルをロード（Renderの無料枠512MBに余裕で収まります！）
model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_FILE = "faq_vectors.json"

if os.path.exists(VECTOR_FILE):
    with open(VECTOR_FILE, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
else:
    faq_data = []

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

@app.get("/")
def read_root():
    return {"message": "Lightweight FAQ Search API is running."}

@app.get("/search")
def search_faq(q: str, limit: int = 3):
    if not q:
        raise HTTPException(status_code=400, detail="Query string 'q' is required.")
    if not faq_data:
        raise HTTPException(status_code=500, detail="FAQ data not loaded.")

    # ユーザーの質問をその場でベクトル化
    query_vector = model.encode(q)

    scored_results = []
    for item in faq_data:
        faq_vector = np.array(item["vector"])
        score = cosine_similarity(query_vector, faq_vector)
        scored_results.append({
            "id": item.get("id"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "score": score
        })

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return {"query": q, "results": scored_results[:limit]}