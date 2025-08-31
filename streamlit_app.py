import streamlit as st
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from api import client, build_bootstrap_user_message, IMAGE_CATALOG, CREATING_DATA_SYSTEM_PROMPT
from move_functions import move_to, pick_object, place_object_next_to, place_object_on, show_room_image, get_room_image_path
from run_and_show import show_provisional_output, run_plan_and_show
from jsonl import save_jsonl_entry, show_jsonl_block

load_dotenv()

ROOM_TOKENS = ["BEDROOM","KITCHEN","DINING","LIVING","BATHROOM","和室","HALL","LDK"]

def detect_rooms_in_text(text: str) -> set[str]:
    found = set()
    up = (text or "").upper()
    for r in ROOM_TOKENS:
        if r == "和室":
            if "和室" in (text or ""):
                found.add(r)
        else:
            if r in up:
                found.add(r)
    return found

def attach_images_for_rooms(rooms: set[str], show_in_ui: bool = True):
    """検出した部屋のうち、まだ送っていない分だけ画像をUI表示＆AIに添付"""
    new_rooms = [r for r in rooms if r not in st.session_state.sent_room_images]
    if not new_rooms:
        return

    local_paths = []
    for room in new_rooms:
        img_path = get_room_image_path(room)  # show_room_image と同じ規則:contentReference[oaicite:5]{index=5}
        if os.path.exists(img_path):
            if show_in_ui:
                show_room_image(room)        # 画面にも表示（任意）:contentReference[oaicite:6]{index=6}
            local_paths.append(img_path)
            st.session_state.sent_room_images.add(room)
        else:
            st.warning(f"{room} の画像が見つかりません: {img_path}")

    if local_paths:
        st.session_state["context"].append(
            build_bootstrap_user_message(
                text=f"Here are room images for: {', '.join(new_rooms)}. Use them for scene understanding and disambiguation.",
                local_image_paths=local_paths,  # ローカル画像をbase64添付:contentReference[oaicite:7]{index=7}
            )
        )

def app():
    st.title("LLMATCHデモアプリ")

    image_root = "images"
    house_dirs = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
    default_label = "(default)"
    options = [default_label] + house_dirs
    current_house = st.session_state.get("selected_house", "")
    current_label = current_house if current_house else default_label
    selected_label = st.selectbox("想定する家", options, index=options.index(current_label) if current_label in options else 0)
    st.session_state["selected_house"] = "" if selected_label == default_label else selected_label

    # 1) セッションにコンテキストを初期化（systemだけ先に入れて保持）
    if "context" not in st.session_state:
        st.session_state["context"] = [
            {"role": "system", "content": CREATING_DATA_SYSTEM_PROMPT},
        ]

    if "active" not in st.session_state:
        st.session_state.active = True
    if "conv_log" not in st.session_state:
        st.session_state.conv_log = {
            "label": "",
            "clarifying_steps": []
        }

    if "sent_room_images" not in st.session_state:
        st.session_state.sent_room_images = set()

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

            def guess_room_from_instruction(text: str) -> str | None:
                rooms = ["BEDROOM","KITCHEN","DINING","LIVING","BATHROOM","和室","HALL"]
                up = text.upper()
                for r in rooms:
                    if r == "和室":
                        if "和室" in text:
                            return r
                    else:
                        if r in up:
                            return r
                return None

            room = guess_room_from_instruction(instruction)
            if room:
                img_path = get_room_image_path(room)  # show_room_image と同じパス規則:contentReference[oaicite:5]{index=5}
                if os.path.exists(img_path):
                    # UIにも表示（任意）
                    show_room_image(room)             # 既存関数で画像を画面に表示:contentReference[oaicite:6]{index=6}
                    # AIには“この部屋の画像だけ”を添付
                    st.session_state["context"].append(
                        build_bootstrap_user_message(
                            text=f"Here is the latest image for {room}. Use it for scene understanding and disambiguation.",
                            local_image_paths=[img_path],  # ←ローカル画像をbase64で添付:contentReference[oaicite:7]{index=7}
                        )
                    )
                else:
                    st.warning(f"{room} の画像が見つかりません: {img_path}")
            
            # 1) ユーザー発話から部屋名を検出 → 新規なら画像添付
            rooms_from_user = detect_rooms_in_text(instruction)
            attach_images_for_rooms(rooms_from_user)

            # 2) 最初のアシスタント応答を取得（画像を添えた状態で）
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state["context"]
            )
            reply = (response.choices[0].message.content).strip()
            st.session_state["context"].append({"role": "assistant", "content": reply})

            # 3) アシスタント応答からも部屋名を検出 → 新規なら画像添付（次の推論に活かす）
            rooms_from_assistant = detect_rooms_in_text(reply)
            attach_images_for_rooms(rooms_from_assistant)
            save_jsonl_entry("insufficient")


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
        save_jsonl_entry("insufficient")

    # 4) 画面下部に履歴を全表示（systemは省く）
    last_assistant_idx = max((i for i, m in enumerate(context) if m["role"] == "assistant"), default=None)

    for i, msg in enumerate(context):
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

        # 最後のアシスタント直後にボタンを出す（計画があるときのみ）
        if i == last_assistant_idx and "<FunctionSequence>" in msg["content"]:
            run_plan_and_show(msg["content"])
            show_provisional_output(msg["content"])
            st.write("この計画はロボットが実行するのに十分ですか？")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("十分", key=f"enough_{i}"):
                    save_jsonl_entry("sufficient")
                    st.session_state.active = False
            with col2:
                if st.button("不十分", key=f"not_enough_{i}"):
                    clarify_prompt = {
                        "role": "system",
                        "content": "The previous plan was insufficient. Ask a clarifying question to the user to improve it."
                    }
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=context + [clarify_prompt]
                    )
                    question = response.choices[0].message.content.strip()
                    context.append({"role": "assistant", "content": question})
                    rooms_from_assistant = detect_rooms_in_text(question)
                    attach_images_for_rooms(rooms_from_assistant)
                    save_jsonl_entry("insufficient")
                    st.rerun()

            if st.session_state.active == False:
                show_jsonl_block()
                st.warning("会話を終了しました。ありがとうございました！")
                if st.button("会話をリセット", key="reset_conv"):
                    # セッション情報を初期化
                    st.session_state.context = [{"role": "system", "content": CREATING_DATA_SYSTEM_PROMPT}]
                    st.session_state.active = True
                    st.session_state.conv_log = {
                        "label": "",
                        "clarifying_steps": []
                    }
                    st.session_state.saved_jsonl = []
                    st.rerun()
                st.stop()

app()
