import streamlit as st
import json
import re
from pathlib import Path
import joblib
import os
from typing import List, Tuple
from dotenv import load_dotenv
from firebase_utils import save_document

load_dotenv()

DATASET_PATH = Path(__file__).parent / "json" / "critic_dataset_train.jsonl"
MODEL_PATH = Path(__file__).parent / "models" / "critic_model_20250903_053907.joblib"
PRE_EXPERIMENT_PATH = Path(__file__).parent / "json" / "pre_experiment_results.jsonl"
EXPERIMENT_1_PATH = Path(__file__).parent / "json" / "experiment_1_results.jsonl"
EXPERIMENT_2_PATH = Path(__file__).parent / "json" / "experiment_2_results.jsonl"


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


def _save_to_firestore(entry, collection_override=None):
    collection = collection_override or os.getenv("FIREBASE_COLLECTION")
    credentials = os.getenv("FIREBASE_CREDENTIALS")
    if not collection:
        print("[Firestore] skipped: no collection name")
        return
    if not credentials:
        print("[Firestore] skipped: no FIREBASE_CREDENTIALS")
        return
    if not Path(credentials).exists():
        print(f"[Firestore] skipped: credentials file not found at {credentials}")
        return
    try:
        save_document(collection, entry, credentials)
        print(f"[Firestore] saved to {collection}")
    except Exception as e:
        # Streamlit なら st.error でも良い
        print(f"[Firestore] ERROR saving to {collection}: {e}")
        raise

def save_jsonl_entry(label: str):
    """会話ログをjsonl形式で1行保存"""
    instruction = next((m["content"] for m in st.session_state.context if m["role"] == "user"), "")
    last_assistant = next((m["content"] for m in reversed(st.session_state.context) if m["role"] == "assistant"), "")
    fs_match = re.search(r"<FunctionSequence>([\s\S]*?)</FunctionSequence>", last_assistant, re.IGNORECASE)
    info_match = re.search(r"<Information>([\s\S]*?)</Information>", last_assistant, re.IGNORECASE)
    function_sequence = fs_match.group(1).strip() if fs_match else ""
    information = info_match.group(1).strip() if info_match else ""

    entry = {
        "instruction": instruction,
        "function_sequence": function_sequence,
        "information": information,
        "label": label,
    }
    if "saved_jsonl" not in st.session_state:
        st.session_state.saved_jsonl = []
    st.session_state.saved_jsonl.append(entry)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    need_newline = False
    if DATASET_PATH.exists() and DATASET_PATH.stat().st_size > 0:
        with DATASET_PATH.open("rb") as f:
            f.seek(-1, 2)
            need_newline = f.read(1) != b"\n"
    with DATASET_PATH.open("a", encoding="utf-8") as f:
        if need_newline:
            f.write("\n")
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _save_to_firestore(entry, collection_override="critic_dataset")

def predict_with_model():
    """学習済みモデルでラベルを推論"""
    # instruction
    instruction = next((m["content"] for m in st.session_state.context if m["role"] == "user"), "")

    # 最新のassistantメッセージから FunctionSequence と Information を抽出
    last_assistant = next((m["content"] for m in reversed(st.session_state.context) if m["role"] == "assistant"), "")
    fs_match = re.search(r"<FunctionSequence>([\s\S]*?)</FunctionSequence>", last_assistant, re.IGNORECASE)
    info_match = re.search(r"<Information>([\s\S]*?)</Information>", last_assistant, re.IGNORECASE)
    function_sequence = fs_match.group(1).strip() if fs_match else ""
    information = info_match.group(1).strip() if info_match else ""

    text = f"instruction: {instruction} \nfs: {function_sequence} \ninfo: {information}"
    model_path = Path(st.session_state.get("model_path", MODEL_PATH))
    model = joblib.load(model_path)
    pred = model.predict([text])[0]
    label = "sufficient" if pred == 1 else "insufficient"

    entry = {
        "instruction": instruction,
        "function_sequence": function_sequence,
        "information": information,
        "label": label,
        "mode": st.session_state.get("mode", "")
    }
    if "saved_jsonl" not in st.session_state:
        st.session_state.saved_jsonl = []
    st.session_state.saved_jsonl.append(entry)

    # DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    # need_newline = False
    # if DATASET_PATH.exists() and DATASET_PATH.stat().st_size > 0:
    #     with DATASET_PATH.open("rb") as f:
    #         f.seek(-1, 2)
    #         need_newline = f.read(1) != b"\n"
    # with DATASET_PATH.open("a", encoding="utf-8") as f:
    #     if need_newline:
    #         f.write("\n")
    #     f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _save_to_firestore(entry, collection_override="predict_with_model")
    return label

def save_pre_experiment_result(human_score: int):
    """保存済みコンテキストから実験結果をjsonl形式で保存"""
    instruction = next((m["content"] for m in st.session_state.context if m["role"] == "user"), "")
    last_assistant = next((m["content"] for m in reversed(st.session_state.context) if m["role"] == "assistant"), "")

    fs_match = re.search(r"<FunctionSequence>([\s\S]*?)</FunctionSequence>", last_assistant, re.IGNORECASE)
    info_match = re.search(r"<Information>([\s\S]*?)</Information>", last_assistant, re.IGNORECASE)

    function_sequence = fs_match.group(1).strip() if fs_match else ""
    information = info_match.group(1).strip() if info_match else ""

    clarifications = []
    user_answers = []
    for m in st.session_state.context:
        if m["role"] == "assistant":
            q_match = re.search(r"<ClarifyingQuestion>([\s\S]*?)</ClarifyingQuestion>", m["content"], re.IGNORECASE)
            if q_match:
                clarifications.append(q_match.group(1).strip())
        if m["role"] == "user":
            content = m["content"]
            if isinstance(content, list):
                # Join list items into a single string
                content = " ".join(map(str, content))
            user_answers.append(content.strip())
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

    clarifications = []
    user_answers = []
    for m in st.session_state.context:
        if m["role"] == "assistant":
            q_match = re.search(r"<ClarifyingQuestion>([\s\S]*?)</ClarifyingQuestion>", m["content"], re.IGNORECASE)
            if q_match:
                clarifications.append(q_match.group(1).strip())
        if m["role"] == "user":
            content = m["content"]
            if isinstance(content, list):
                # Join list items into a single string
                content = " ".join(map(str, content))
            user_answers.append(content.strip())
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
        "function_sequence": function_sequence,
        "information": information,
        "clarification_question": clarifications,
        # TODO: user_answersは保存しないか、image_urlを除いて保存するか考える
        # "user_answers": user_answers,
        "similarity": similarity,
        "human_scores": human_scores,
        "mode": st.session_state.get("mode", ""),
        "function_count": function_count,
        "function_variable_lengths": variable_lengths,
    }
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

    clarifications = []
    user_answers = []
    for m in st.session_state.context:
        if m["role"] == "assistant":
            q_match = re.search(r"<ClarifyingQuestion>([\s\S]*?)</ClarifyingQuestion>", m["content"], re.IGNORECASE)
            if q_match:
                clarifications.append(q_match.group(1).strip())
        if m["role"] == "user":
            content = m["content"]
            if isinstance(content, list):
                # Join list items into a single string
                content = " ".join(map(str, content))
            user_answers.append(content.strip())
    text = f"instruction: {instruction} \nfs: {function_sequence}"

    entry = {
        "instruction": instruction,
        "function_sequence": function_sequence,
        "information": information,
        "clarification_question": clarifications,
        # TODO: user_answersは保存しないか、image_urlを除いて保存するか考える
        # "user_answers": user_answers,
        "human_scores": human_scores,
        "prompt_label": st.session_state.get("prompt_label", ""),
        "termination_label": termination_label,
        "termination_reason": termination_reason,
        "function_count": function_count,
        "function_variable_lengths": variable_lengths,
    }

    if termination_label:
        entry["termination_label"] = termination_label
    if termination_reason:
        entry["termination_reason"] = termination_reason

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
