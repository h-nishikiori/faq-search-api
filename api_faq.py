import os
import numpy as np
import json
import requests
from fastapi import FastAPI, HTTPException

app = FastAPI()

# 🔑 ご共有いただいた最新の無料APIキー
GEMINI_API_KEY = "AQ.Ab8RN6LtOjtZWXjPZzOQPSuCK001_6jCE7EY8Bac33XdO0aLPg"

# 💡 404エラーを100%回避する、正式な「v1」のWeb通信用URL
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"

VECTOR_FILE = "faq_vectors.json"

if os.path.exists(VECTOR_FILE):
    with open(VECTOR_FILE, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
else:
    faq_data = []

# コサイン類似度計算（安全版）
def cosine_similarity(v1, v2):
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    
    # 💡 データの次元数（数字の個数）が一致しているか念のためチェック
    if v1.shape != v2.shape:
        return 0.0
        
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

@app.get("/")
def read_root():
    return {"message": "Gemini Ultra-lightweight FAQ Search API is running."}

@app.get("/search")
def search_faq(q: str, limit: int = 3):
    if not q:
        raise HTTPException(status_code=400, detail="Query string 'q' is required.")
    if not faq_data:
        raise HTTPException(status_code=500, detail="FAQ vector data is not loaded.")

    # ユーザーからの質問をGeminiに送りベクトル化
    payload = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": q}]
        },
        "taskType": "RETRIEVAL_QUERY"
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(GEMINI_URL, json=payload, headers=headers)
        res_json = response.json()
        
        if "embedding" in res_json:
            query_vector = res_json["embedding"]["values"]
        else:
            raise HTTPException(status_code=500, detail=f"Gemini API Error: {res_json}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Communication Error: {str(e)}")

    # 類似度計算
    scored_results = []
    for item in faq_data:
        # 💡 jsonから読み込んだベクトルを確実にnumpy配列に変換
        if "vector" in item and item["vector"] is not None:
            faq_vector = item["vector"]
            score = cosine_similarity(query_vector, faq_vector)
        else:
            score = 0.0
            
        scored_results.append({
            "id": item.get("id"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "score": score
        })

    # スコア順にソート（降順）
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return {"query": q, "results": scored_results[:limit]}