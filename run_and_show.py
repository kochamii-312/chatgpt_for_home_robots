import streamlit as st
import re
from strips import parse_step
from move_functions import move_to, pick_object, place_object_next_to, place_object_on, detect_object

FUNCTION_DOCS = """
- **move_to(room_name:str)**
  指定した部屋へロボットを移動します。

- **pick_object(object:str)**
  指定した物体をつかみます。

- **place_object_next_to(object:str, target:str)**
  指定した物体をターゲットの横に置きます。

- **place_object_on(object:str, target:str)**
  指定した物体をターゲットの上に置きます。

- **place_object_in(object:str, target:str)**
  指定した物体をターゲットの中に入れます。

- **detect_object(object:str)**
  指定した物体を検出します。

- **search_about(object:str)**
  指定した物体に関する情報を検索します。

- **push(object:str)**
  指定した物体を押します。

- **say(text:str)**
  指定したテキストを発話します。
"""

def show_function_sequence(reply: str):
    """<FunctionSequence> ... </FunctionSequence> をコードブロックで表示"""
    func_match = re.search(r"<FunctionSequence>([\s\S]*?)</FunctionSequence>", reply, re.IGNORECASE)
    if not func_match:
        return
    st.markdown("#### ロボット行動計画")
    st.code(func_match.group(0), language="xml")
    with st.expander("行動計画で使用される関数", expanded=False):
        st.markdown(FUNCTION_DOCS)

def show_clarifying_question(reply: str):
    """<ClarifyingQuestion> ... </ClarifyingQuestion> を通常のテキストで表示"""
    q_match = re.search(
        r"<ClarifyingQuestion>([\s\S]*?)</ClarifyingQuestion>",
        reply,
        re.IGNORECASE,
    )
    if q_match:
        question_text = q_match.group(1)
    else:
        fallback_match = re.search(r"<ClarifyingQuestion>([\s\S]*)", reply, re.IGNORECASE)
        if not fallback_match:
            return
        question_text = fallback_match.group(1)
    st.markdown("#### ロボットからの質問")
    st.write(question_text.strip())


def show_information(reply: str):
    """<Information> ... </Information> を蓄積して表示"""
    info_match = re.search(r"<Information>([\s\S]*?)</Information>", reply, re.IGNORECASE)
    if not info_match:
        return

    items = re.findall(r"<li>(.*?)</li>", info_match.group(1))
    if "information_items" not in st.session_state:
        st.session_state.information_items = []
    for item in items:
        if item not in st.session_state.information_items:
            st.session_state.information_items.append(item)

    aggregated = "".join(f"<li>{item}</li>" for item in st.session_state.information_items)
    st.subheader("Information")
    st.markdown("<ul>" + aggregated + "</ul>", unsafe_allow_html=True)


def show_provisional_output(reply: str):
    """<ProvisionalOutput> 内の関数列と確認質問のみを表示"""
    prov_match = re.search(
        r"<ProvisionalOutput>([\s\S]*?)</ProvisionalOutput>", reply, re.IGNORECASE
    )
    if not prov_match:
        return
    provisional = prov_match.group(1)
    show_function_sequence(provisional)
    show_clarifying_question(provisional)

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
