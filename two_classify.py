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
DEFAULT_TRAIN_PATH = DATA_DIR / "critic_dataset_train.json"
DEFAULT_VALID_PATH = DATA_DIR / "critic_dataset_valid.json"


def load_jsonl(path: str | Path) -> list[dict]:
    """jsonlファイルを読み込み、空行を除外して辞書のリストを返す。"""

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(path: str | Path) -> list[dict]:
    """訓練・検証用データセットを読み込む。

    デフォルトではjson形式のファイルを想定するが、後方互換のために
    jsonl形式のファイルが存在する場合はそちらも利用できるようにする。
    """

    path = Path(path)

    if path.exists():
        if path.suffix == ".jsonl":
            return load_jsonl(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # list以外の形式で保存されていた場合でも安全に扱う
            return list(data or [])

    legacy_path = path.with_suffix(".jsonl")
    if legacy_path.exists():
        return load_jsonl(legacy_path)

    return []


def prepare_data(data: Iterable[dict]) -> tuple[list[str], list[int]]:
    """学習・推論で利用するテキストとラベルを作成する。"""

    texts: list[str] = []
    labels: list[int] = []
    for ex in data:
        parts: list[str] = []

        instruction = (ex.get("instruction") or "").strip()
        if instruction:
            parts.append(f"Instruction: {instruction}")

        function_sequence = (ex.get("function_sequence") or "").strip()
        if function_sequence:
            parts.append(f"FunctionSequence: {function_sequence}")

        clarifying_history = ex.get("clarifying_history") or []
        history_segments: list[str] = []
        for step in clarifying_history:
            if isinstance(step, dict):
                question = (
                    step.get("clarifying_question")
                    or step.get("question")
                    or step.get("llm_question")
                    or ""
                ).strip()
                answer = (
                    step.get("chat_input")
                    or step.get("user_answer")
                    or step.get("answer")
                    or ""
                ).strip()
                pair_parts = []
                if question:
                    pair_parts.append(f"Q: {question}")
                if answer:
                    pair_parts.append(f"A: {answer}")
                if pair_parts:
                    history_segments.append(" ".join(pair_parts))
            elif step:
                history_segments.append(str(step))
        if history_segments:
            parts.append("ClarifyingHistory: " + " || ".join(history_segments))

        information = (ex.get("information") or "").strip()
        if information:
            parts.append(f"Information: {information}")

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

    train_data = load_dataset(train_path)
    valid_data = load_dataset(valid_path)

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
