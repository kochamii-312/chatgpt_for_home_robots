import streamlit as st
import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "json" / "critic_dataset_train.jsonl"

def save_jsonl_entry(label: str):
    """会話ログをjsonl形式で1行保存"""
    # instruction
    instruction = next((m["content"] for m in st.session_state.context if m["role"] == "user"), "")
    # clarifying_steps: assistant→userのペア
    clarifying_steps = []
    msgs = st.session_state.context
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == "assistant" and msgs[i+1]["role"] == "user":
            clarifying_steps.append({
                "llm_question": msgs[i]["content"],
                "user_answer": msgs[i+1]["content"]
            })
    entry = {
        "instruction": instruction,
        "clarifying_steps": clarifying_steps,
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

def show_jsonl_block():
    """保存済みjsonlデータをコードブロックで表示"""
    if "saved_jsonl" in st.session_state and st.session_state.saved_jsonl:
        st.subheader("保存済みJSONLデータ")
        jsonl_str = "\n".join([json.dumps(e, ensure_ascii=False) for e in st.session_state.saved_jsonl])
        st.code(jsonl_str, language="json")
