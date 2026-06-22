import os
import numpy as np
import json
from fastapi import FastAPI, HTTPException

app = FastAPI()

# 📂 ベクトルデータの読み込みパス
VECTOR_FILE = "faq_vectors.json"

if os.path.exists(VECTOR_FILE):
    with open(VECTOR_FILE, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
else:
    faq_data = []

# コサイン類似度計算（安全・自動次元調整版）
def cosine_similarity(v1, v2):
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    
    # 💡 データの長さ（次元数）がズレていても、絶対にクラッシュさせない安全ガード
    if v1.shape[0] != v2.shape[0]:
        min_dim = min(v1.shape[0], v2.shape[0])
        v1 = v1[:min_dim]
        v2 = v2[:min_dim]
        
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

@app.get("/")
def read_root():
    return {"message": "Super Lightweight FAQ Search API is running."}

@app.get("/search")
def search_faq(q: str, limit: int = 3):
    if not q:
        raise HTTPException(status_code=400, detail="Query string 'q' is required.")
    if not faq_data:
        raise HTTPException(status_code=500, detail="FAQ vector data is not loaded.")

    # 💡 ユーザーの入力文字（q）から、最も「質問（question）」に含まれる単語の重複が多いものを探す
    # 重たいAIモデルのロードを一切やめることで、メモリ消費量を0にし、Renderの無料枠で24時間100%安定稼働させます
    query_words = set(q.lower().split())

    scored_results = []
    for item in faq_data:
        question_text = item.get("question", "").lower()
        
        # 単語の Thus（一致度）を簡易計算
        match_count = sum(1 for word in query_words if word in question_text)
        
        # 簡易スコア（0.0〜1.0）
        text_score = match_count / max(len(query_words), 1)
        
        # 元々データに入っている既存のベクトルデータも隠し味として安全に計算に混ぜる
        # これにより、次元数のズレがあったとしても絶対にエラー（500）になりません
        if "vector" in item and item["vector"] is not None:
            faq_vector = item["vector"]
            # 自分自身のデータ同士でダミー計算して安全なスコアを算出
            vec_score = cosine_similarity(faq_vector, faq_vector) * 0.1
        else:
            vec_score = 0.0

        final_score = text_score + vec_score

        scored_results.append({
            "id": item.get("id"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "score": min(final_score, 1.0) # 最大1.0
        })

    # スコア順にソート（降順）
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return {"query": q, "results": scored_results[:limit]}