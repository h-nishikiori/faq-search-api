import json
import numpy as np
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer

# 1. FastAPIの初期化
app = FastAPI(title="FAQ Vector Search API")

# 2. サーバー起動時に、AIモデルとベクトルデータをメモリに常駐（常時スタンバイ状態にする）
print("--- [START] Loading AI Model and FAQ Vectors ---")
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("faq_vectors.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)
print("--- [SUCCESS] Ready for Search Requests ---")


# 3. コサイン類似度の計算関数
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


# 4. 検索を受け付けるエンドポイント ( http://127.0.0.1:8000/search?q=検索ワード )
@app.get("/search")
def search_faq(q: str = Query(..., description="検索キーワード（英語の文章など）")):
    # ユーザーの入力文をベクトル化
    query_vector = model.encode(q)
    
    results = []
    for item in faq_data:
        if "vector" in item:
            similarity = cosine_similarity(query_vector, item["vector"])
            results.append({
                "id": item["id"],          # PHP側での判別用（CSVの行番号ベース）
                "question": item["question"],
                "score": similarity
            })
            
    # スコアが高い順にソート
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    # 上位3件の「IDとスコアと質問」だけを、PHPが処理しやすいJSON形式で返却
    return {
        "query": q,
        "results": results[:3]
    }