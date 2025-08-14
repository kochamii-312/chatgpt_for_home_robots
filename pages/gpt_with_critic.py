import streamlit as st
from openai import OpenAI
import re
import json
from dotenv import load_dotenv
from api import client, SYSTEM_PROMPT, move_to, pick_object, place_object
import joblib
from two_classify import prepare_data  # 既存関数を利用

load_dotenv()

TAG_RE = re.compile(r"</?([A-Za-z0-9_]+)(\s[^>]*)?>")

def strip_tags(text: str) -> str:
    return TAG_RE.sub("", text or "").strip()

def extract_between(tag: str, text: str) -> str | None:
    m = re.search(fr"<{tag}>([\s\S]*?)</{tag}>", text or "", re.IGNORECASE)
    return m.group(1).strip() if m else None

def run_plan_and_show(reply: str):
    """<Plan> ... </Plan> を見つけて実行し、結果を表示"""
    plan_match = re.search(r"<Plan>(.*?)</Plan>", reply, re.S)
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

def finalize_and_render_plan(label: str):
    """会話終了時に行動計画をまとめて画面表示"""
    # final_answer の決定
    last_assistant = next((m for m in reversed(st.session_state.context) if m["role"] == "assistant"), None)
    final_answer = extract_between("FinalAnswer", last_assistant["content"]) if last_assistant else None
    if not final_answer and last_assistant:
        final_answer = strip_tags(last_assistant["content"])

    st.session_state.conv_log["final_answer"] = final_answer or ""
    st.session_state.conv_log["label"] = "sufficient" if label == "sufficient" else "insufficient"

    # question_label が None のステップは継続が無ければ insufficient で埋める
    for s in st.session_state.conv_log["clarifying_steps"]:
        if s["question_label"] is None:
            s["question_label"] = "insufficient"

    st.subheader("会話サマリ（JSON）")
    st.code(
        json.dumps(st.session_state.conv_log, ensure_ascii=False, indent=2),
        language="json"
    )

# 事前に two_classify.py で学習済みモデルを保存しておく（例: joblib.dump(model, "critic_model.joblib")）
model = joblib.load("critic_model.joblib")

def get_critic_label(context):
    # contextから判定用テキストを生成
    instruction = next((m["content"] for m in context if m["role"] == "user"), "")
    clarifying_steps = []
    for m in context:
        if m["role"] == "assistant":
            q = extract_between("llm_question", m["content"]) or ""
            a = extract_between("user_answer", m["content"]) or ""
            if q and a:
                clarifying_steps.append({"llm_question": q, "user_answer": a})
    ex = {"instruction": instruction, "clarifying_steps": clarifying_steps, "label": "unknown"}
    texts, _ = prepare_data([ex])
    pred = model.predict(texts)
    return "sufficient" if pred[0] == 1 else "insufficient"

def app():
    st.title("LLMATCHデモアプリ")
    st.subheader("ChatGPT with 'Critic'")

    # 1) セッションにコンテキストを初期化（systemだけ先に入れて保持）
    if "context" not in st.session_state:
        st.session_state["context"] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if "active" not in st.session_state:
        st.session_state.active = True
    if "conv_log" not in st.session_state:
        st.session_state.conv_log = {
            "final_answer": "",
            "label": "",
            "clarifying_steps": []
        }

    context = st.session_state["context"]

    message = st.chat_message("assistant")
    message.write("こんにちは、私は家庭用ロボットです！あなたの指示に従って行動します。")
    user_input = st.chat_input("ロボットへの指示や回答を入力してください")
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

    # sufficient判定なら終了
    label = get_critic_label(context)
    if label == "sufficient":
        st.success("クリティックモデルが「十分」と判定したため会話を終了します。")
        finalize_and_render_plan(label="sufficient")
        st.stop()

    # 画面下部に履歴を全表示（systemは省く）
    # iが20になったら会話終了
    if len(context) - sum(1 for m in context if m["role"] == "system") >= 20:
        st.success("会話が20ターンに達したため終了します。")
        finalize_and_render_plan(label="sufficient")  # 必要に応じてラベルを変更
        st.stop()

    for i, msg in enumerate(context):
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


app()
