import streamlit as st
import json
import re
from pathlib import Path
import joblib

DATASET_PATH = Path(__file__).parent / "json" / "critic_dataset_train.jsonl"
MODEL_PATH = Path(__file__).parent / "critic_model.joblib"

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
        "label": label
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

def save_jsonl_entry_with_model():
    """会話ログをjsonl形式で1行保存し、学習済みモデルでラベルを推論"""
    # instruction
    instruction = next((m["content"] for m in st.session_state.context if m["role"] == "user"), "")

    # 最新のassistantメッセージから FunctionSequence と FinalOutput を抽出
    last_assistant = next((m["content"] for m in reversed(st.session_state.context) if m["role"] == "assistant"), "")
    fs_match = re.search(r"<FunctionSequence>([\s\S]*?)</FunctionSequence>", last_assistant, re.IGNORECASE)
    fo_match = re.search(r"<FinalOutput>([\s\S]*?)</FinalOutput>", last_assistant, re.IGNORECASE)
    function_sequence = fs_match.group(1).strip() if fs_match else ""
    final_output = fo_match.group(1).strip() if fo_match else ""

    text = f"instruction: {instruction} \nfs: {function_sequence} \nfo: {final_output}"
    model = joblib.load(MODEL_PATH)
    pred = model.predict([text])[0]
    label = "sufficient" if pred == 1 else "insufficient"

    entry = {
        "instruction": instruction,
        "function_sequence": function_sequence,
        "final_output": final_output,
        "label": label
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

    return label

def show_jsonl_block():
    """保存済みjsonlデータをコードブロックで表示"""
    if "saved_jsonl" in st.session_state and st.session_state.saved_jsonl:
        st.subheader("保存済みJSONLデータ")
        jsonl_str = "\n".join([json.dumps(e, ensure_ascii=False) for e in st.session_state.saved_jsonl])
        st.code(jsonl_str, language="json")
