import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# データ読み込み
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl("critic_dataset_train.jsonl")
valid_data = load_jsonl("critic_dataset_valid.jsonl")

# テキストとラベルを作成
def prepare_data(data):
    texts = []
    labels = []
    for ex in data:
        clarifying_text = " ".join([f"Q:{s['llm_question']} A:{s['user_answer']}" for s in ex["clarifying_steps"]])
        text = f"指示: {ex['instruction']} || {clarifying_text}"
        texts.append(text)
        labels.append(1 if ex["label"] == "sufficient" else 0)
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
