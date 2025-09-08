import json
import joblib
import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split

# データ読み込み
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        # 空行を除外して読み込む
        return [json.loads(line) for line in f if line.strip()]

train_data = load_jsonl("./json/critic_dataset_train.jsonl")
valid_data = load_jsonl("./json/critic_dataset_valid.jsonl")

# テキストとラベルを作成
def prepare_data(data):
    texts = []
    labels = []
    for ex in data:
        # instruction, function_sequence, information を連結して特徴とする
        parts = [ex.get("instruction", ""),
                 ex.get("function_sequence", ""),
                 ex.get("information", "")]
        text = " | ".join(parts)
        texts.append(text)
        labels.append(1 if ex.get("label") == "sufficient" else 0)
    return texts, labels

all_data = train_data + valid_data  # 全部まとめる
X_all, y_all = prepare_data(all_data)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# 1) クラス重み + 2) n-gram/min_df を追加
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)),
    ('clf', LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# 学習
model.fit(X_train, y_train)


# 4) 閾値最適化（検証データでF1が最大の閾値を選ぶ）
proba = model.predict_proba(X_valid)[:, 1]
prec, rec, th = precision_recall_curve(y_valid, proba)
f1 = (2*prec*rec)/(prec+rec+1e-9)
best_idx = np.argmax(f1)
best_th = th[best_idx] if best_idx < len(th) else 0.5  # 念のため
y_pred_opt = (proba >= best_th).astype(int)

print(f"Best threshold ≈ {best_th:.3f}  (valid F1={f1[best_idx]:.3f})")
print(classification_report(y_valid, y_pred_opt, zero_division=0))

# 評価
pred = model.predict(X_valid)
print(classification_report(y_valid, pred))

# モデル保存（タイムスタンプ付き）
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"models/critic_model_{timestamp}.joblib"
joblib.dump(model, filename)
print(f"モデルを保存しました: {filename}")