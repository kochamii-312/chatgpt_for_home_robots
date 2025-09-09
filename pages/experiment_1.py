import streamlit as st
import json
import os
from difflib import SequenceMatcher
from move_functions import (
    move_to,
    pick_object,
    place_object_next_to,
    place_object_on,
    show_room_image,
    get_room_image_path,
)
from dotenv import load_dotenv
from api import client, SYSTEM_PROMPT
from strips import strip_tags, extract_between
from run_and_show import (
    show_function_sequence,
    show_clarifying_question,
    run_plan_and_show,
)
from jsonl import save_jsonl_entry_with_model, save_experiment_1_result
from run_and_show import show_provisional_output
from room_utils import detect_rooms_in_text, attach_images_for_rooms
from pathlib import Path

load_dotenv()

def app():
    st.title("LLMATCHデモアプリ")
    st.subheader("実験① GPTとGPT with Criticの比較")

    mode_options = ["GPT", "GPT with critic"]
    default_mode = st.session_state.get("mode", "GPT with critic")
    mode = st.radio("モード選択", mode_options, index=mode_options.index(default_mode), horizontal=True)
    st.session_state["mode"] = mode

    system_prompt = SYSTEM_PROMPT

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
        st.session_state.turn_count = 0
    if "active" not in st.session_state:
        st.session_state.active = True
    if "sent_room_images" not in st.session_state:
        st.session_state.sent_room_images = set()
    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0

    context = st.session_state["context"]

    message = st.chat_message("assistant")
    message.write("こんにちは、私は家庭用ロボットです！あなたの指示に従って行動します。")
    input_box = st.chat_input("ロボットへの回答を入力してください")
    user_input = st.session_state.pop("pending_user_input", None) or input_box
    
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
        st.session_state.turn_count += 1
        
    # 画面下部に履歴を全表示（systemは省く）
    last_assistant_idx = max((i for i, m in enumerate(context) if m["role"] == "assistant"), default=None)
    
    for i, msg in enumerate(context):
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                if i == last_assistant_idx and "<FunctionSequence>" in msg["content"]:
                    run_plan_and_show(msg["content"])
                show_function_sequence(msg["content"])
                show_clarifying_question(msg["content"])
    label = save_jsonl_entry_with_model()
    should_stop = False
    end_message = ""
    if st.session_state.get("mode") == "GPT with critic":
        if label == "sufficient":
            should_stop = True
            end_message = "モデルがsufficientを出力したため終了します。"
    else:
        if st.session_state.turn_count >= 7:
            should_stop = True
            end_message = "7回の会話に達したため終了します。"

    if should_stop:
        st.success(end_message)
        if st.session_state.active:
            with st.form("evaluation_form"):
                feasibility = st.radio(
                    "使う関数は適切か（不要なものが含まれている / 違う関数の方が適切）（1-4）",
                    [1, 2, 3, 4],
                    horizontal=True,
                )
                variables = st.radio(
                    "関数の変数は適切か（間違ったオブジェクトが入っている / もっと良い変数がある）（1-4）",
                    [1, 2, 3, 4],
                    horizontal=True,
                )
                specificity = st.radio(
                    "関数の変数の具体性（1-4）",
                    [1, 2, 3, 4],
                    horizontal=True,
                )
                hallucination = st.radio(
                    "実際にはないもの・伝えていない情報を含めていないか（1-4）",
                    [1, 2, 3, 4],
                    horizontal=True,
                )
                coverage = st.radio(
                    "聞いたことがすべて盛り込まれているか（1-4）",
                    [1, 2, 3, 4],
                    horizontal=True,
                )
                obstacle = st.radio(
                    "障害物があれば、避けられるか（1-4）",
                    [1, 2, 3, 4],
                    horizontal=True,
                )
                selection = st.radio(
                    "複数のものがある中で適切なものが選べるか（1-4）",
                    [1, 2, 3, 4],
                    horizontal=True,
                )
                extra_question = st.radio(
                    "会話の中で余計な質問・不自然な質問があったか（1-4）",
                    [1, 2, 3, 4],
                    horizontal=True,
                )
                submitted = st.form_submit_button("評価を保存")

            if submitted:
                scores = {
                    "feasibility": feasibility,
                    "variables": variables,
                    "specificity": specificity,
                    "hallucination": hallucination,
                    "coverage": coverage,
                    "obstacle": obstacle,
                    "selection": selection,
                    "extra_question": extra_question,
                }
                save_experiment_1_result(scores)
                st.session_state.active = False

        if st.session_state.active == False:
            st.warning("会話を終了しました。ありがとうございました！")
            if st.button("会話をリセット", key="reset_conv"):
                st.session_state.context = [{"role": "system", "content": system_prompt}]
                st.session_state.active = True
                st.session_state.conv_log = {
                    "label": "",
                    "clarifying_steps": []
                }
                st.session_state.saved_jsonl = []
                st.session_state.turn_count = 0
                st.rerun()
            st.stop()
    
    if st.button("会話をリセット", key="reset_conv"):
        st.session_state.context = [{"role": "system", "content": system_prompt}]
        st.session_state.active = True
        st.session_state.conv_log = {
            "label": "",
            "clarifying_steps": []
        }
        st.session_state.saved_jsonl = []
        st.session_state.turn_count = 0
        st.rerun()

app()
