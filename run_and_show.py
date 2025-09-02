import streamlit as st
import re
from strips import parse_step
from move_functions import move_to, pick_object, place_object_next_to, place_object_on, show_room_image, detect_object


def show_function_sequence(reply: str):
    """<FunctionSequence> ... </FunctionSequence> をコードブロックで表示"""
    func_match = re.search(r"<FunctionSequence>([\s\S]*?)</FunctionSequence>", reply, re.IGNORECASE)
    if not func_match:
        return
    st.subheader("Function sequence")
    st.code(func_match.group(0), language="xml")


def show_clarifying_question(reply: str):
    """<ClarifyingQuestion> ... </ClarifyingQuestion> を通常のテキストで表示"""
    q_match = re.search(r"<ClarifyingQuestion>([\s\S]*?)</ClarifyingQuestion>", reply, re.IGNORECASE)
    if not q_match:
        return
    st.subheader("Clarifying question")
    st.write(q_match.group(1).strip())

def run_plan_and_show(reply: str):
    """<FunctionSequence> を見つけて実行し、結果を表示"""
    func_match = re.search(r"<FunctionSequence>(.*?)</FunctionSequence>", reply, re.S)
    if not func_match:
        return
    steps = re.findall(r"<Updated>(.*?)</Updated>", func_match.group(1))
    if not steps:
        return

    with st.expander("Plan 実行ログ", expanded=True):
        for step in steps:
            try:
                py_step = parse_step(step)
                result = eval(py_step)
                st.write(f"✅ `{py_step}` → **{result}**")
            except Exception as e:
                st.write(f"⚠️ `{step}` の実行でエラー: {e}")
