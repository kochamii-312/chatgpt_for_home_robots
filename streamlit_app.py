import streamlit as st
import json
from openai import OpenAI
from dotenv import load_dotenv
from api import client, build_bootstrap_user_message, IMAGE_CATALOG, CREATING_DATA_SYSTEM_PROMPT
from move_functions import move_to, pick_object, place_object_next_to, place_object_on, show_room_image
from run_and_show import show_provisional_output, run_plan_and_show
from jsonl import save_jsonl_entry, show_jsonl_block

load_dotenv()

def app():
    st.title("LLMATCHデモアプリ")

    # 1) セッションにコンテキストを初期化（systemだけ先に入れて保持）
    if "context" not in st.session_state:
        bootstrap_text = (
            "Attached are reference images for the environment. "
            "Please use them during reasoning. "
            "When the plan moves to a room (e.g., KITCHEN, BATHROOM, LDK), "
            "refer to the corresponding image in conversation."
        )

        # 画像カタログのすべてのURLをリストにまとめる
        all_urls = [url for url in IMAGE_CATALOG.values() if url]

        st.session_state["context"] = [
            {"role": "system", "content": CREATING_DATA_SYSTEM_PROMPT},
            build_bootstrap_user_message(
                text=bootstrap_text,
                image_urls=all_urls,
            ),
        ]

        # 「全部送るとトークンがもったいない」「部屋が多すぎる」場合は、ユーザーが『KITCHENへ行け』と入力したタイミングで、その部屋画像を user メッセージとして追加する
        # ユーザー指示に "KITCHEN" が含まれていたら、そのとき画像を追加
        # if "KITCHEN" in instruction.upper():
        #     context.append(build_bootstrap_user_message(
        #         text="Here is the KITCHEN image for reference.",
        #         image_urls=[IMAGE_CATALOG["KITCHEN"]],
        #     ))

    if "active" not in st.session_state:
        st.session_state.active = True
    if "conv_log" not in st.session_state:
        st.session_state.conv_log = {
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

    # 4) 画面下部に履歴を全表示（systemは省く）
    last_assistant_idx = max((i for i, m in enumerate(context) if m["role"] == "assistant"), default=None)

    for i, msg in enumerate(context):
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

        # 最後のアシスタント直後にボタンを出す
        if i == last_assistant_idx:
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
                    save_jsonl_entry("insufficient")
                    st.success("jsonl形式でデータを1行保存しました！")

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
