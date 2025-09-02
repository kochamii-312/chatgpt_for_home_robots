import streamlit as st
from openai import OpenAI
import re
import json
import os
from move_functions import move_to, pick_object, place_object_next_to, place_object_on, show_room_image, get_room_image_path
from dotenv import load_dotenv
from api import (
    client,
    SYSTEM_PROMPT_STANDARD,
    SYSTEM_PROMPT_FRIENDLY,
    SYSTEM_PROMPT_PRATFALL,
)
from strips import strip_tags, extract_between
from run_and_show import show_function_sequence, show_clarifying_question, run_plan_and_show
from jsonl import save_jsonl_entry_with_model
from room_utils import detect_rooms_in_text, attach_images_for_rooms

load_dotenv()

def show_provisional_output(reply: str):
    show_function_sequence(reply)
    show_clarifying_question(reply)
    run_plan_and_show(reply)

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

def app():
    st.title("LLMATCHデモアプリ")
    st.subheader("プレ実験")
    st.write("目的：GPT with Criticの学習の効果を図る。")
    st.write("定量的評価：人間が作った行動計画の正解と、対話によって最終的に生成されたロボットの行動計画を比較し、どれくらい一致するかを検証する。")
    st.write("定性的評価：対話によって最終的に生成されたロボットの行動計画が実行可能かを評価する。")

    prompt_options = {
        "Standard": SYSTEM_PROMPT_STANDARD,
        "Friendly": SYSTEM_PROMPT_FRIENDLY,
        "Pratfall": SYSTEM_PROMPT_PRATFALL,
    }
    prompt_label = st.selectbox("プロンプト", list(prompt_options.keys()))
    system_prompt = prompt_options[prompt_label]

    image_root = "images"
    house_dirs = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
    default_label = "(default)"
    options = [default_label] + house_dirs
    current_house = st.session_state.get("selected_house", "")
    current_label = current_house if current_house else default_label
    selected_label = st.selectbox("想定する家", options, index=options.index(current_label) if current_label in options else 0)
    st.session_state["selected_house"] = "" if selected_label == default_label else selected_label

    # 1) セッションにコンテキストを初期化（systemだけ先に入れて保持）
    if (
        "context" not in st.session_state
        or st.session_state.get("system_prompt") != system_prompt
    ):
        st.session_state["context"] = [{"role": "system", "content": system_prompt}]
        st.session_state["system_prompt"] = system_prompt
        st.session_state.conv_log = {
            "final_answer": "",
            "label": "",
            "clarifying_steps": []
        }
    if "active" not in st.session_state:
        st.session_state.active = True
    if "sent_room_images" not in st.session_state:
        st.session_state.sent_room_images = set()

    context = st.session_state["context"]

    message = st.chat_message("assistant")
    message.write("こんにちは、私は家庭用ロボットです！あなたの指示に従って行動します。")
    user_input = st.chat_input("ロボットへの回答を入力してください")
    
    if user_input:
        context.append({"role": "user", "content": user_input})
        rooms_from_user = detect_rooms_in_text(user_input)
        attach_images_for_rooms(rooms_from_user)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context
        )
        reply = response.choices[0].message.content.strip()
        print("Assistant:", reply)
        context.append({"role": "assistant", "content": reply})
        rooms_from_assistant = detect_rooms_in_text(reply)
        attach_images_for_rooms(rooms_from_assistant)
        print("context: ", context)
        label = save_jsonl_entry_with_model()
        if label == "sufficient":
            st.success("モデルがsufficientを出力したため終了します。")
            finalize_and_render_plan(label)
            st.stop()

    last_assistant_idx = max((i for i, m in enumerate(context) if m["role"] == "assistant"), default=None)
        
    # 画面下部に履歴を全表示（systemは省く）

    for i, msg in enumerate(context):
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
        
        if i == last_assistant_idx:
            show_provisional_output(msg["content"])
    if st.button("会話をリセット", key="reset_conv"):
        # セッション情報を初期化
        st.session_state.context = [{"role": "system", "content": system_prompt}]
        st.session_state.active = True
        st.session_state.conv_log = {
            "label": "",
            "clarifying_steps": []
        }
        st.session_state.saved_jsonl = []
        st.rerun()

app()
