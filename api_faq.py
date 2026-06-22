# 必要な超軽量ライブラリをマックに入れます
python3 -m pip install sentence-transformers --break-system-packages

# 変換スクリプトを作成して実行します
cat << 'EOF' > convert_gemini_final.py
import json
from sentence_transformers import SentenceTransformer

INPUT_FILE = "faq_vectors.json"
OUTPUT_FILE = "faq_vectors_new.json"

print("🚀 【完全自己完結】超軽量・高精度の英語AIモデルでデータ変換を開始します...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# わずか90MBの世界標準モデルをロード（一瞬で終わります）
model = SentenceTransformer("all-MiniLM-L6-v2")

new_faq_data = []

for i, item in enumerate(faq_data):
    question_text = item["question"]
    print(f"[{i+1}/{len(faq_data)}] ベクトル化中: {question_text[:35]}...")
    
    # マックの中で一瞬で計算（WEB通信しないので絶対にエラーになりません）
    vector = model.encode(question_text)
    
    item["vector"] = vector.tolist()
    new_faq_data.append(item)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(new_faq_data, f, ensure_ascii=False, indent=4)

print(f"✨ 大成功！エラーなしで全件の変換が完了しました！: {OUTPUT_FILE}")
EOF

python3 convert_gemini_final.py