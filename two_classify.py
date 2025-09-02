import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

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
        labels.append(1 if ex.get["label"] == "sufficient" else 0)
    return texts, labels

X_train, y_train = prepare_data(train_data)
X_valid, y_valid = prepare_data(valid_data)

# モデル作成
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=200))
])

# 学習
model.fit(X_train, y_train)

# 評価
pred = model.predict(X_valid)
print(classification_report(y_valid, pred))

# モデル保存
joblib.dump(model, "critic_model.joblib")