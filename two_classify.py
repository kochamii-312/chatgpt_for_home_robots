import datetime
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_DIR = Path(__file__).parent / "json"
DEFAULT_TRAIN_PATH = DATA_DIR / "critic_dataset_train.jsonl"
DEFAULT_VALID_PATH = DATA_DIR / "critic_dataset_valid.jsonl"


def load_jsonl(path: str | Path) -> list[dict]:
    """jsonlファイルを読み込み、空行を除外して辞書のリストを返す。"""

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_data(data: Iterable[dict]) -> tuple[list[str], list[int]]:
    """学習・推論で利用するテキストとラベルを作成する。"""

    texts: list[str] = []
    labels: list[int] = []
    for ex in data:
        parts = [
            ex.get("instruction", ""),
            ex.get("function_sequence", ""),
            ex.get("information", ""),
        ]
        text = " | ".join(parts)
        texts.append(text)
        labels.append(1 if ex.get("label") == "sufficient" else 0)
    return texts, labels


def build_pipeline() -> Pipeline:
    """分類モデルの推論パイプラインを生成する。"""

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_and_save_model(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    valid_path: str | Path = DEFAULT_VALID_PATH,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    """データセットを読み込みモデルを学習し、joblib形式で保存する。"""

    train_data = load_jsonl(train_path)
    valid_data = load_jsonl(valid_path)

    all_data: list[dict] = list(train_data) + list(valid_data)
    X_all, y_all = prepare_data(all_data)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_valid)[:, 1]
    prec, rec, th = precision_recall_curve(y_valid, proba)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1))
    best_th = th[best_idx] if best_idx < len(th) else 0.5
    y_pred_opt = (proba >= best_th).astype(int)

    print(f"Best threshold ≈ {best_th:.3f}  (valid F1={f1[best_idx]:.3f})")
    print(classification_report(y_valid, y_pred_opt, zero_division=0))

    pred = model.predict(X_valid)
    print(classification_report(y_valid, pred))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path("models") / f"critic_model_{timestamp}.joblib"
    filename.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filename)
    print(f"モデルを保存しました: {filename}")

    return filename


if __name__ == "__main__":
    train_and_save_model()
