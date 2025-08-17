import streamlit as st
import re
from strips import parse_step

def show_provisional_output(reply: str):
    """
    <ProvisionalOutput> ... </ProvisionalOutput> をコードブロックで表示
    """
    # Provisional output の抽出と表示
    prov_match = re.search(r"<ProvisionalOutput>([\s\S]*?)</ProvisionalOutput>", reply, re.IGNORECASE)
    if not prov_match:
        return
    st.subheader("Provisional output")
    st.code(prov_match.group(0), language="xml")

def run_plan_and_show(reply: str):
    """
    <FunctionSequence> ... </FunctionSequence> を見つけて実行し、結果を表示
    """
    # FunctionSequence の抽出と実行
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
