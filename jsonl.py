import json
import os
import re
from itertools import zip_longest
from pathlib import Path

import joblib
import streamlit as st

from typing import Any, List, Optional, Tuple
from dotenv import load_dotenv
from firebase_utils import save_document
from api import client

load_dotenv()

DATASET_PATH = Path(__file__).parent / "json" / "critic_dataset_train.json"
LEGACY_DATASET_PATH = DATASET_PATH.with_suffix(".jsonl")
MODEL_PATH = Path(__file__).parent / "models" / "critic_model_20250903_053907.joblib"
PRE_EXPERIMENT_PATH = Path(__file__).parent / "json" / "pre_experiment_results.jsonl"
EXPERIMENT_1_PATH = Path(__file__).parent / "json" / "experiment_1_results.jsonl"
EXPERIMENT_2_PATH = Path(__file__).parent / "json" / "experiment_2_results.jsonl"

PLAN_SUCCESS_PROMPT_TEMPLATE = """Here is a home robot task instruction and the resulting action plan.
Evaluate the **probability of success (0–100%)** if this plan were executed.

- "Success" means the plan matches the user’s intent, is feasible, safe, and logically consistent.
- Base your judgment only on the provided instruction and function sequence.
- Output only the number (e.g., 75). Do not include any explanation.

[Instruction]
{instruction}

[Available Functions]
<Functions>
    <Function name="move_to" args="room_name:str">Move robot to the specified room.</Function>
    <Function name="pick_object" args="object:str">Pick up the specified object.</Function>
    <Function name="place_object_next_to" args="object:str, target:str">Place the object next to the target.</Function>
    <Function name="place_object_on" args="object:str, target:str">Place the object on the target.</Function>
    <Function name="place_object_in" args="object:str, target:str">Place the object in the target.</Function>
    <Function name="detect_object" args="object:str">Detect the specified object using YOLO.</Function>
    <Function name="search_about" args="object:str">Search information about the specified object.</Function>
    <Function name="push" args="object:str">Push the specified object.</Function>
    <Function name="say" args="text:str">Speak the specified text.</Function>
</Functions>

[Function Sequence]
{function_sequence}
"""


def _extract_clarifying_question(text: str) -> Optional[str]:
    """Extract clarifying question text even if the closing tag is missing."""
    if not isinstance(text, str):
        return None
    match = re.search(
        r"<ClarifyingQuestion>([\s\S]*?)</ClarifyingQuestion>",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    fallback_match = re.search(r"<ClarifyingQuestion>([\s\S]*)", text, re.IGNORECASE)
    if fallback_match:
        return fallback_match.group(1).strip()
    return None


def _analyze_function_sequence(function_sequence: str) -> Tuple[int, List[int]]:
    """関数数と各関数の変数文字数を取得する"""

    if not function_sequence:
        return 0, []

    function_pattern = re.compile(r"<\s*(?!/)([\w_]+)([^>]*)>")
    attr_pattern = re.compile(r"=\s*\"([^\"]*)\"")

    function_count = 0
    variable_lengths: List[int] = []

    for match in function_pattern.finditer(function_sequence):
        tag_name = match.group(1)
        if tag_name.lower() == "functionsequence":
            continue
        attributes = match.group(2) or ""
        matched_text = match.group(0)
        is_self_closing = matched_text.rstrip().endswith("/>")

        values = [v.strip() for v in attr_pattern.findall(attributes)]

        inner_text = ""
        if not is_self_closing:
            closing_tag = f"</{tag_name}>"
            start_index = match.end()
            end_index = function_sequence.find(closing_tag, start_index)
            if end_index != -1:
                inner_text = function_sequence[start_index:end_index]
                inner_text = inner_text.strip()

        if inner_text:
            values.append(inner_text)

        total_length = sum(len(value) for value in values)

        function_count += 1
        variable_lengths.append(total_length)

    return function_count, variable_lengths


def evaluate_plan_success_probability(
    instruction: Optional[str],
    function_sequence: Optional[str],
) -> Optional[float]:
    instruction = (instruction or "").strip()
    function_sequence = (function_sequence or "").strip()
    if not instruction or not function_sequence:
        return None

    prompt = PLAN_SUCCESS_PROMPT_TEMPLATE.format(
        instruction=instruction,
        function_sequence=function_sequence,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"[PlanEval] Failed to call evaluation model: {e}")
        return None

    result_text = response.choices[0].message.content.strip()
    match = re.search(r"\d+(?:\.\d+)?", result_text)
    if not match:
        print(f"[PlanEval] Unexpected response: {result_text}")
        return None

    try:
        value = float(match.group(0))
    except ValueError:
        print(f"[PlanEval] Unable to parse probability from response: {result_text}")
        return None

    return max(0.0, min(100.0, value))


def _save_to_firestore(entry, collection_override=None):
    collection = collection_override or os.getenv("FIREBASE_COLLECTION")
    creds = (
        os.getenv("FIREBASE_CREDENTIALS")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # ← 追加: ADC/パス
        or None
    )
    if not collection:
        print("[Firestore] skipped: no collection name")
        return
    try:
        # creds が None でも、save_document 側で ADC を使える実装なら通る
        save_document(collection, entry, creds)
        print(f"[Firestore] saved to {collection}")
    except Exception as e:
        print(f"[Firestore] ERROR saving to {collection}: {e}")
        raise

def _load_jsonl_entries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_dataset_entries() -> list[dict]:
    if DATASET_PATH.exists():
        try:
            with DATASET_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
        if isinstance(data, list):
            return data
        return list(data or [])

    if LEGACY_DATASET_PATH.exists():
        return _load_jsonl_entries(LEGACY_DATASET_PATH)

    return []


def _save_dataset_entries(entries: list[dict]) -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _collect_clarifying_history() -> list[dict[str, str]]:
    clarifying_questions: list[str] = []
    for message in st.session_state.get("context", []):
        if message.get("role") != "assistant":
            continue
        question = _extract_clarifying_question(message.get("content", ""))
        if question:
            clarifying_questions.append(question)

    chat_inputs = [
        ans.strip()
        for ans in st.session_state.get("chat_input_history", [])
        if ans and ans.strip()
    ]

    clarifying_history: list[dict[str, str]] = []
    for question, user_input in zip_longest(
        clarifying_questions, chat_inputs, fillvalue=""
    ):
        question = question.strip() if isinstance(question, str) else ""
        user_input = user_input.strip() if isinstance(user_input, str) else ""
        if not question and not user_input:
            continue
        clarifying_history.append(
            {
                "clarifying_question": question,
                "chat_input": user_input,
            }
        )

    return clarifying_history


def _format_clarifying_history_for_model(
    clarifying_history: list[dict[str, str]] | list
) -> str:
    segments: list[str] = []
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
            pair: list[str] = []
            if question:
                pair.append(f"Q: {question}")
            if answer:
                pair.append(f"A: {answer}")
            if pair:
                segments.append(" ".join(pair))
        elif step:
            segments.append(str(step))
    return " || ".join(segments)


def build_critic_model_input(
    instruction: str,
    function_sequence: str,
    clarifying_history: list[dict[str, str]] | list,
    information: str,
) -> str:
    parts: list[str] = []

    instruction = (instruction or "").strip()
    if instruction:
        parts.append(f"Instruction: {instruction}")

    function_sequence = (function_sequence or "").strip()
    if function_sequence:
        parts.append(f"FunctionSequence: {function_sequence}")

    history_text = _format_clarifying_history_for_model(clarifying_history or [])
    if history_text:
        parts.append(f"ClarifyingHistory: {history_text}")

    information = (information or "").strip()
    if information:
        parts.append(f"Information: {information}")

    return " | ".join(parts)


def remove_last_jsonl_entry():
    """Remove the last saved entry from the dataset file and session cache."""

    if "saved_jsonl" in st.session_state and st.session_state.saved_jsonl:
        st.session_state.saved_jsonl.pop()

    entries = _load_dataset_entries()
    if not entries:
        return

    entries.pop()
    _save_dataset_entries(entries)


def save_jsonl_entry(label: str):
    """会話ログをデータセットファイルへ保存"""
    instruction = next((m["content"] for m in st.session_state.context if m["role"] == "user"), "")
    last_assistant = next((m["content"] for m in reversed(st.session_state.context) if m["role"] == "assistant"), "")
    fs_match = re.search(r"<FunctionSequence>([\s\S]*?)</FunctionSequence>", last_assistant, re.IGNORECASE)
    info_match = re.search(r"<Information>([\s\S]*?)</Information>", last_assistant, re.IGNORECASE)
    function_sequence = fs_match.group(1).strip() if fs_match else ""
    information = info_match.group(1).strip() if info_match else ""

    clarifying_history = _collect_clarifying_history()

    entry = {
        "instruction": instruction,
        "function_sequence": function_sequence,
        "clarifying_history": clarifying_history,
        "information": information,
        "label": label,
    }
    if "saved_jsonl" not in st.session_state:
        st.session_state.saved_jsonl = []
    st.session_state.saved_jsonl.append(entry)

    entries = _load_dataset_entries()
    entries.append(entry)
    _save_dataset_entries(entries)
    _save_to_firestore(entry, collection_override="critic_dataset")

def predict_with_model():
    """学習済みモデルでラベルを推論（有効しきい値を使用）"""
    # instruction
    instruction = next((m["content"] for m in st.session_state.context if m["role"] == "user"), "")

    # 最新assistantから FS / Information
    last_assistant = next((m["content"] for m in reversed(st.session_state.context) if m["role"] == "assistant"), "")
    fs_match = re.search(r"<FunctionSequence>([\s\S]*?)</FunctionSequence>", last_assistant, re.IGNORECASE)
    info_match = re.search(r"<Information>([\s\S]*?)</Information>", last_assistant, re.IGNORECASE)
    function_sequence = fs_match.group(1).strip() if fs_match else ""
    information = info_match.group(1).strip() if info_match else ""

    clarifying_history = _collect_clarifying_history()
    text = build_critic_model_input(
        instruction,
        function_sequence,
        clarifying_history,
        information,
    )

    # モデル+しきい値を一度だけロード
    model_path = Path(st.session_state.get("model_path", MODEL_PATH))
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        model = obj.get("model", obj)
        saved_th = float(obj.get("threshold", 0.5))
    else:
        model = obj
        saved_th = 0.5  # 後方互換

    # 確率
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba([text])[0, 1])
    elif hasattr(model, "decision_function"):
        import numpy as np
        z = model.decision_function([text])[0]
        p = float(1 / (1 + np.exp(-z)))
    else:
        p = float(model.predict([text])[0])  # 最低限のフォールバック

    th_min  = float(st.session_state.get("critic_min_threshold", 0.60))
    force   = st.session_state.get("critic_force_threshold", None)
    if force is not None:
        th_eff = float(force)                 # ← 強制上書き
    else:
        th_eff = max(saved_th, th_min)        # ← 従来の下限フロア

    label = "sufficient" if p >= th_eff else "insufficient"

    # 保存・ログ（saved_th も入れておく）
    entry = {
        "instruction": instruction,
        "function_sequence": function_sequence,
        "clarifying_history": clarifying_history,
        "information": information,
        "mode": st.session_state.get("mode", ""),
        "prediction": label,
        "probability": p,
        "threshold_eff": th_eff,
        "threshold_saved": saved_th,
    }
    if "saved_jsonl" not in st.session_state:
        st.session_state.saved_jsonl = []
    st.session_state.saved_jsonl.append(entry)
    _save_to_firestore(entry, collection_override="predict_with_model")

    # UI/呼び出し側がそのまま使えるよう、有効しきい値を返す
    return label, p, th_eff

def save_pre_experiment_result(human_score: int):
    """保存済みコンテキストから実験結果をjsonl形式で保存"""
    instruction = next((m["content"] for m in st.session_state.context if m["role"] == "user"), "")
    last_assistant = next((m["content"] for m in reversed(st.session_state.context) if m["role"] == "assistant"), "")

    fs_match = re.search(r"<FunctionSequence>([\s\S]*?)</FunctionSequence>", last_assistant, re.IGNORECASE)
    info_match = re.search(r"<Information>([\s\S]*?)</Information>", last_assistant, re.IGNORECASE)

    function_sequence = fs_match.group(1).strip() if fs_match else ""
    information = info_match.group(1).strip() if info_match else ""

    clarifying_history = _collect_clarifying_history()
    text = f"instruction: {instruction} \nfs: {function_sequence}"
    similarity = None
    model_path = Path(st.session_state.get("model_path", MODEL_PATH))
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            similarity = float(model.predict_proba([text])[0][1])
        except Exception:
            similarity = None

    entry = {
        "instruction": instruction,
        "function_sequence": function_sequence,
        "information": information,
        "clarification_question": clarifications,
        "user_answers": user_answers,
        "similarity": similarity,
        "human_score": human_score,
        "mode": st.session_state.get("mode", "")
    }

    success_probability = evaluate_plan_success_probability(
        instruction,
        function_sequence,
    )
    if success_probability is not None:
        entry["plan_success_probability"] = success_probability

    if "saved_jsonl" not in st.session_state:
        st.session_state.saved_jsonl = []
    st.session_state.saved_jsonl.append(entry)

    PRE_EXPERIMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    need_newline = False
    if PRE_EXPERIMENT_PATH.exists() and PRE_EXPERIMENT_PATH.stat().st_size > 0:
        with PRE_EXPERIMENT_PATH.open("rb") as f:
            f.seek(-1, 2)
            need_newline = f.read(1) != b"\n"
    with PRE_EXPERIMENT_PATH.open("a", encoding="utf-8") as f:
        if need_newline:
            f.write("\n")
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _save_to_firestore(entry, collection_override="pre_experiment_results")

def save_experiment_1_result(
    human_scores: dict,
    termination_reason: str = "",
    termination_label: str = "",
):
    """保存済みコンテキストから実験結果をjsonl形式で保存

    Parameters
    ----------
    human_scores: dict
        4段階評価の結果を格納した辞書。各質問項目をキー、評価を値として渡す。
    """
    instruction = next((m["content"] for m in st.session_state.context if m["role"] == "user"), "")
    last_assistant = next((m["content"] for m in reversed(st.session_state.context) if m["role"] == "assistant"), "")

    fs_match = re.search(r"<FunctionSequence>([\s\S]*?)</FunctionSequence>", last_assistant, re.IGNORECASE)
    info_match = re.search(r"<Information>([\s\S]*?)</Information>", last_assistant, re.IGNORECASE)

    function_sequence = fs_match.group(1).strip() if fs_match else ""
    information = info_match.group(1).strip() if info_match else ""

    function_count, variable_lengths = _analyze_function_sequence(function_sequence)

    human_scores = dict(human_scores)
    success_probability = evaluate_plan_success_probability(
        instruction,
        function_sequence,
    )
    if success_probability is not None:
        human_scores["plan_success_probability"] = success_probability

    clarifying_history = _collect_clarifying_history()
    text = f"instruction: {instruction} \nfs: {function_sequence}"
    # TODO: 類似度どうするか考える。プレ実験にしか含めないか、experiment_1にも含めるか
    similarity = None
    model_path = Path(st.session_state.get("model_path", MODEL_PATH))
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            similarity = float(model.predict_proba([text])[0][1])
        except Exception:
            similarity = None

    entry = {
        "instruction": instruction,
        "information": information,
        "clarifying_history": clarifying_history,
        "similarity": similarity,
        "human_scores": human_scores,
        "mode": st.session_state.get("mode", ""),
        "function": {
            "sequence": function_sequence,
            "count": function_count,
            "variable_lengths": variable_lengths,
        },
    }
    if success_probability is not None:
        entry["plan_success_probability"] = success_probability
    if termination_label:
        entry["termination_label"] = termination_label
    if termination_reason:
        entry["termination_reason"] = termination_reason

    if "saved_jsonl" not in st.session_state:
        st.session_state.saved_jsonl = []
    st.session_state.saved_jsonl.append(entry)

    EXPERIMENT_1_PATH.parent.mkdir(parents=True, exist_ok=True)
    need_newline = False
    if EXPERIMENT_1_PATH.exists() and EXPERIMENT_1_PATH.stat().st_size > 0:
        with EXPERIMENT_1_PATH.open("rb") as f:
            f.seek(-1, 2)
            need_newline = f.read(1) != b"\n"
    with EXPERIMENT_1_PATH.open("a", encoding="utf-8") as f:
        if need_newline:
            f.write("\n")
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _save_to_firestore(entry, collection_override="experiment_1_results")


def _strip_visible_text(text: Optional[str]) -> str:
    """Convert assistant output into the plain text shown to users."""

    if not text:
        return ""

    cleaned = re.sub(r"</li>\s*", "\n", text)
    cleaned = re.sub(r"<li>\s*", "- ", cleaned)
    cleaned = re.sub(r"</?([A-Za-z0-9_]+)(\s[^>]*)?>", "", cleaned)
    return cleaned.strip()


def save_experiment_2_result(
    human_scores: dict,
    termination_reason: str = "",
    termination_label: str = "",
):
    """保存済みコンテキストから実験結果をjsonl形式で保存

    Parameters
    ----------
    human_scores: dict
        4段階評価の結果を格納した辞書。各質問項目をキー、評価を値として渡す。
    """
    instruction = next((m["content"] for m in st.session_state.context if m["role"] == "user"), "")
    last_assistant = next((m["content"] for m in reversed(st.session_state.context) if m["role"] == "assistant"), "")

    fs_match = re.search(r"<FunctionSequence>([\s\S]*?)</FunctionSequence>", last_assistant, re.IGNORECASE)
    info_match = re.search(r"<Information>([\s\S]*?)</Information>", last_assistant, re.IGNORECASE)

    function_sequence = fs_match.group(1).strip() if fs_match else ""
    information = info_match.group(1).strip() if info_match else ""

    function_count, variable_lengths = _analyze_function_sequence(function_sequence)

    human_scores = dict(human_scores)
    success_probability = evaluate_plan_success_probability(
        instruction,
        function_sequence,
    )
    if success_probability is not None:
        human_scores["plan_success_probability"] = success_probability

    clarifying_history = _collect_clarifying_history()
    text = f"instruction: {instruction} \nfs: {function_sequence}"

    assistant_visible_messages = [
        visible
        for visible in (
            _strip_visible_text(m.get("content", ""))
            for m in st.session_state.context
            if m.get("role") == "assistant"
        )
        if visible
    ]

    entry = {
        "instruction": instruction,
        "information": information,
        "clarifying_history": clarifying_history,
        "human_scores": human_scores,
        "prompt_label": st.session_state.get("prompt_label", ""),
        "termination_label": termination_label,
        "termination_reason": termination_reason,
        "function": {
            "sequence": function_sequence,
            "count": function_count,
            "variable_lengths": variable_lengths,
        },
    }
    if success_probability is not None:
        entry["plan_success_probability"] = success_probability

    if termination_label:
        entry["termination_label"] = termination_label
    if termination_reason:
        entry["termination_reason"] = termination_reason

    if assistant_visible_messages:
        entry["assistant_visible_messages"] = assistant_visible_messages

    if "saved_jsonl" not in st.session_state:
        st.session_state.saved_jsonl = []
    st.session_state.saved_jsonl.append(entry)

    EXPERIMENT_2_PATH.parent.mkdir(parents=True, exist_ok=True)
    need_newline = False
    if EXPERIMENT_2_PATH.exists() and EXPERIMENT_2_PATH.stat().st_size > 0:
        with EXPERIMENT_2_PATH.open("rb") as f:
            f.seek(-1, 2)
            need_newline = f.read(1) != b"\n"
    with EXPERIMENT_2_PATH.open("a", encoding="utf-8") as f:
        if need_newline:
            f.write("\n")
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _save_to_firestore(entry, collection_override="experiment_2_results")

def show_jsonl_block():
    """保存済みjsonlデータをコードブロックで表示"""
    if "saved_jsonl" in st.session_state and st.session_state.saved_jsonl:
        st.subheader("保存済みJSONLデータ")
        jsonl_str = "\n".join([json.dumps(e, ensure_ascii=False) for e in st.session_state.saved_jsonl])
        st.code(jsonl_str, language="json")
