import streamlit as st
import json
import re
from pathlib import Path
import joblib
import os

from typing import Any, List, Optional, Tuple
from dotenv import load_dotenv
from firebase_utils import save_document
from api import client

load_dotenv()

DATASET_PATH = Path(__file__).parent / "json" / "critic_dataset_train.jsonl"
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


def remove_last_jsonl_entry():
    """Remove the last saved entry from the dataset file and session cache."""

    if "saved_jsonl" in st.session_state and st.session_state.saved_jsonl:
        st.session_state.saved_jsonl.pop()

    if not DATASET_PATH.exists():
        return

    with DATASET_PATH.open("rb+") as f:
        f.seek(0, os.SEEK_END)
        file_end = f.tell()
        if file_end == 0:
            return

        pos = file_end - 1

        # Skip trailing newlines at the end of the file
        while pos >= 0:
            f.seek(pos)
            if f.read(1) != b"\n":
                break
            pos -= 1

        if pos < 0:
            f.truncate(0)
            return

        # Find the newline that precedes the last line and truncate after it
        while pos >= 0:
            f.seek(pos)
            if f.read(1) == b"\n":
                f.truncate(pos + 1)
                return
            pos -= 1

        # No newline found, meaning there was a single line
        f.truncate(0)


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
    if success_probability is not None:
        entry["plan_success_probability"] = success_probability

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
