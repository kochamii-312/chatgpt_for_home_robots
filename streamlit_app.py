import streamlit as st
from openai import OpenAI
import re
import json
from dotenv import load_dotenv
from api import client, CREATING_DATA_SYSTEM_PROMPT, move_to, pick_object, place_object

load_dotenv()

TAG_RE = re.compile(r"</?([A-Za-z0-9_]+)(\s[^>]*)?>")

def strip_tags(text: str) -> str:
    return TAG_RE.sub("", text or "").strip()

def extract_between(tag: str, text: str) -> str | None:
    m = re.search(fr"<{tag}>([\s\S]*?)</{tag}>", text or "", re.IGNORECASE)
    return m.group(1).strip() if m else None

def run_plan_and_show(reply: str):
    """<Plan> ... </Plan> を見つけて実行し、結果を表示
       <Provisional output> ... </Provosional output> をコードブロックで表示
    """
    # Provisional output の抽出と表示
    prov_match = re.search(r"<Provisional output>([\s\S]*?)</Provosional output>", reply, re.IGNORECASE)
    if prov_match:
        st.subheader("Provisional output")
        st.code(prov_match.group(0), language="xml")

    # Plan の抽出と実行
    plan_match = re.search(r"<Sequence of function>(.*?)</Sequence of function>", reply, re.S)
    if not plan_match:
        return
    steps = re.findall(r"<Step>(.*?)</Step>", plan_match.group(1))
    if not steps:
        return

    with st.expander("Plan 実行ログ", expanded=True):
        for step in steps:
            try:
                result = eval(step)  # 例: move_to(1.0, 2.0)
                st.write(f"✅ `{step}` → **{result}**")
            except Exception as e:
                st.write(f"⚠️ `{step}` の実行でエラー: {e}")

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

def show_jsonl_block():
    """保存済みjsonlデータをコードブロックで表示"""
    if "saved_jsonl" in st.session_state and st.session_state.saved_jsonl:
        st.subheader("保存済みJSONLデータ")
        jsonl_str = "\n".join([json.dumps(e, ensure_ascii=False) for e in st.session_state.saved_jsonl])
        st.code(jsonl_str, language="json")

def app():
    st.title("LLMATCHデモアプリ")

    # 1) セッションにコンテキストを初期化（systemだけ先に入れて保持）
    if "context" not in st.session_state:
        st.session_state["context"] = [{"role": "system", "content": CREATING_DATA_SYSTEM_PROMPT}]
    if "active" not in st.session_state:
        st.session_state.active = True
    if "conv_log" not in st.session_state:
        st.session_state.conv_log = {
            "final_answer": "",
            "label": "",
            "clarifying_steps": []
        }

    context = st.session_state["context"]

    # 2) フォーム：ここで送信したら即時に最初の応答まで取得して表示
    with st.form(key="instruction_form"):
        st.subheader("ロボットへの指示")
        instruction = st.text_input("ロボットへの指示")
        submit_btn = st.form_submit_button("実行")

    if submit_btn:
        if not instruction.strip():
            st.warning("指示が空です。内容を入力してください。")
        else:
            # フォーム送信のタイミングでユーザー指示を表示
            st.success(f"ロボットへの指示がセットされました：**{instruction}**")
            context.append({"role": "user", "content": instruction})

            # 最初のアシスタント応答を取得
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=context
            )
            reply = (response.choices[0].message.content).strip()
            print("Assistant:", reply)
            context.append({"role": "assistant", "content": reply})

            # 可能なら Plan を実行
            run_plan_and_show(reply)

    # 3) 追加の自由入力（会話継続用）
    user_input = st.chat_input("入力してください")
    if user_input:
        context.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context
        )
        reply = response.choices[0].message.content.strip()
        print("Assistant:", reply)
        context.append({"role": "assistant", "content": reply})
        print("context: ", context)

        run_plan_and_show(reply)

    # 4) 画面下部に履歴を全表示（systemは省く）
    last_assistant_idx = max((i for i, m in enumerate(context) if m["role"] == "assistant"), default=None)

    for i, msg in enumerate(context):
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

        # 最後のアシスタント直後にボタンを出す
        if i == last_assistant_idx:
            st.write("この計画はロボットが実行するのに十分ですか？")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("十分", key=f"enough_{i}"):
                    save_jsonl_entry("sufficient")
                    st.session_state.active = False
            with col2:
                if st.button("不十分", key=f"not_enough_{i}"):
                    save_jsonl_entry("insufficient")
                    st.success("jsonl形式でデータを1行保存しました！")

            if st.session_state.active == False:
                show_jsonl_block()
                st.warning("会話を終了しました。ありがとうございました！")
                st.stop()

app()
