import streamlit as st
import json
import os

from dotenv import load_dotenv

from api import client, SYSTEM_PROMPT, build_bootstrap_user_message
from jsonl import predict_with_model, save_experiment_1_result
from move_functions import (
    move_to,
    pick_object,
    place_object_next_to,
    place_object_on,
)
from run_and_show import (
    run_plan_and_show,
    show_clarifying_question,
    show_function_sequence,
)
from run_and_show import show_provisional_output
from strips import extract_between, strip_tags
from tasks.ui import render_random_room_task

load_dotenv()

def app():
    st.title("LLMATCH Criticデモアプリ")
    st.subheader("実験① GPTとGPT with Criticの比較")
    
    st.sidebar.subheader("行動計画で使用される関数")
    st.sidebar.markdown(
    """
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
    )

    mode_options = ["GPT", "GPT with critic"]
    default_mode = st.session_state.get("mode", "GPT with critic")
    mode = st.radio("モード選択", mode_options, index=mode_options.index(default_mode), horizontal=True)
    st.session_state["mode"] = mode

    system_prompt = SYSTEM_PROMPT

    model_files = sorted(
        f for f in os.listdir("models") if f.endswith(".joblib")
    )
    if model_files:
        latest_model = max(
            model_files,
            key=lambda f: os.path.getmtime(os.path.join("models", f)),
        )
        stored_model = st.session_state.get("model_path")
        current_model = os.path.basename(stored_model) if stored_model else None
        if current_model not in model_files:
            current_model = latest_model
        selected_model = st.selectbox(
            "評価モデル",
            model_files,
            index=model_files.index(current_model),
        )
        st.session_state["model_path"] = os.path.join("models", selected_model)

    image_root = "images"
    house_dirs = [d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))]
    default_label = "(default)"
    options = [default_label] + house_dirs
    current_house = st.session_state.get("selected_house", "")
    current_label = current_house if current_house else default_label
    selected_label = st.selectbox(
        "想定する家",
        options,
        index=options.index(current_label) if current_label in options else 0,
    )
    st.session_state["selected_house"] = "" if selected_label == default_label else selected_label

    image_dir = image_root
    subdirs = []
    if st.session_state["selected_house"]:
        image_dir = os.path.join(image_dir, st.session_state["selected_house"])
        subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    sub_default = "(default)"
    if subdirs:
        current_sub = st.session_state.get("selected_subfolder", "")
        current_sub_label = current_sub if current_sub else sub_default
        sub_options = [sub_default] + subdirs
        sub_label = st.selectbox(
            "部屋",
            sub_options,
            index=sub_options.index(current_sub_label) if current_sub_label in sub_options else 0,
        )
        st.session_state["selected_subfolder"] = "" if sub_label == sub_default else sub_label
        if st.session_state["selected_subfolder"]:
            image_dir = os.path.join(image_dir, st.session_state["selected_subfolder"])
    else:
        st.session_state["selected_subfolder"] = ""

    selected_room = st.session_state.get("selected_subfolder", "")
    render_random_room_task(selected_room, state_prefix="experiment1")

    if os.path.isdir(image_dir):
        image_files = [
            f
            for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
        ]
        if image_files:
            selected_imgs = st.multiselect("表示する画像", image_files)
            selected_paths = [os.path.join(image_dir, img) for img in selected_imgs]
            st.session_state["selected_image_paths"] = selected_paths
            for path, img in zip(selected_paths, selected_imgs):
                st.image(path, caption=img)
        else:
            st.session_state["selected_image_paths"] = []

    # 1) セッションにコンテキストを初期化（systemだけ先に入れて保持）
    if (
        "context" not in st.session_state
        or st.session_state.get("system_prompt") != system_prompt
    ):
        st.session_state["context"] = [{"role": "system", "content": system_prompt}]
        st.session_state["system_prompt"] = system_prompt
        st.session_state.conv_log = {
            "information": "",
            "label": "",
            "clarifying_steps": []
        }
        st.session_state.turn_count = 0
        st.session_state["chat_input_history"] = []
    if "chat_input_history" not in st.session_state:
        st.session_state["chat_input_history"] = []
    if "active" not in st.session_state:
        st.session_state.active = True
    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0
    if "force_end" not in st.session_state:
        st.session_state.force_end = False
    if "end_reason" not in st.session_state:
        st.session_state.end_reason = ""

    context = st.session_state["context"]

    message = st.chat_message("assistant")
    message.write("こんにちは、私は家庭用ロボットです！あなたの指示に従って行動します。")
    if st.session_state.get("force_end"):
        user_input = None
    else:
        input_box = st.chat_input("ロボットへの回答を入力してください")
        if input_box:
            st.session_state["chat_input_history"].append(input_box)
        user_input = st.session_state.pop("pending_user_input", None) or input_box
    
    if user_input:
        context.append({"role": "user", "content": user_input})
        selected_paths = st.session_state.get("selected_image_paths", [])
        if selected_paths:
            context.append(
                build_bootstrap_user_message(
                    text="Here are the selected images. Use them for scene understanding and disambiguation.",
                    local_image_paths=selected_paths,
                )
            )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context
        )
        reply = response.choices[0].message.content.strip()
        print("Assistant:", reply)
        context.append({"role": "assistant", "content": reply})
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
    label = predict_with_model()
    should_stop = False
    end_message = ""
    if st.session_state.get("force_end"):
        should_stop = True
        end_message = "ユーザーが会話を強制的に終了しました。"
    elif st.session_state.get("mode") == "GPT with critic":
        if label == "sufficient":
            should_stop = True
            end_message = "モデルがsufficientを出力したため終了します。"
    else:
        if st.session_state.turn_count >= 4:
            should_stop = True
            end_message = "4回の会話に達したため終了します。"

    if should_stop:
        st.success(end_message)
        if st.session_state.active:
            with st.form("evaluation_form"):
                name = st.text_input(
                    "あなたの名前やユーザーネーム等（被験者区別用）"
                )
                success = st.radio(
                    "行動計画が実行されたとして、ロボットは成功しますか？", 
                    ["成功する", "成功しない"], 
                    horizontal=True
                )
                failure_reason = st.multiselect(
                    "成功しない場合、その理由を教えてください。（複数選択可）",
                    [
                        "関数が不適切・不足している",
                        "変数が不適切・具体的でない",
                        "虚偽の情報が含まれている",
                        "会話の中で出てきた必要な情報を含んでいない",
                        "複数のものがある中で適切なものが選べない",
                        "以上の理由以外", 
                        "成功する"
                    ]
                )
                failure_reason_others = st.text_input(
                    "前の質問で「以上の理由以外」を選んだ方はその内容を書いてください。"
                )
                grices_maxim = st.multiselect(
                    "ロボットの発言に関して、以下の内容の中で当てはまるものがあれば選んでください。（複数選択可）",
                    [
                        "嘘や虚偽の情報を述べた",
                        "質問・情報提供が多すぎるまたは少なすぎる",
                        "タスクを実行するのに関係のない発言があった",
                        "コミュニケーションが明確でなかった（何と答えればいいかわからない質問があった等）"
                    ]
                )
                familiarity = st.radio(
                    "ロボットにどれくらい親近感を持ちましたか？（1-4）",
                    [1, 2, 3, 4],
                    horizontal=True
                )
                social_presence = st.radio(
                    "対話の相手がそこに存在し、自分と同じ空間を共有している、あるいは自分と関わっている感覚（ソーシャルプレゼンス）をどれくらい持ちましたか？（1-4）",
                    [1, 2, 3, 4],
                    horizontal=True
                )
                free = st.text_input(
                    "その他に何か感じたことがあればお願いします。"
                )
                submitted = st.form_submit_button("評価を保存")

            if submitted:
                scores = {
                    "name": name,
                    "success": success,
                    "failure_reason": failure_reason,
                    "failure_reason_others": failure_reason_others,
                    "grices_maxim": grices_maxim,
                    "familiarity": familiarity,
                    "social_presence": social_presence,
                    "free": free,
                }
                termination_label = "会話を強制的に終了" if st.session_state.get("force_end") else ""
                save_experiment_1_result(
                    scores,
                    st.session_state.get("end_reason", ""),
                    termination_label,
                )
                st.session_state.active = False

        if st.session_state.active == False:
            st.warning("会話を終了しました。ありがとうございました！")
            cols_end = st.columns([1, 1, 2])
            with cols_end[0]:
                if st.button("⚠️会話をリセット", key="reset_conv_end"):
                    st.session_state.context = [{"role": "system", "content": system_prompt}]
                    st.session_state.active = True
                    st.session_state.conv_log = {
                        "label": "",
                        "clarifying_steps": []
                    }
                    st.session_state.saved_jsonl = []
                    st.session_state.turn_count = 0
                    st.session_state.force_end = False
                    st.session_state.end_reason = ""
                    st.session_state["chat_input_history"] = []
                    st.rerun()
            with cols_end[1]:
                st.button("🚨会話を強制的に終了", key="force_end_disabled", disabled=True)
            with cols_end[2]:
                st.text_input("会話を終了したい理由", key="end_reason", disabled=True)
            st.stop()

    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("⚠️会話をリセット", key="reset_conv"):
            st.session_state.context = [{"role": "system", "content": system_prompt}]
            st.session_state.active = True
            st.session_state.conv_log = {
                "label": "",
                "clarifying_steps": []
            }
            st.session_state.saved_jsonl = []
            st.session_state.turn_count = 0
            st.session_state.force_end = False
            st.session_state.end_reason = ""
            st.session_state["chat_input_history"] = []
            st.rerun()
    with cols[1]:
        if st.button("🚨会話を強制的に終了", key="force_end_button"):
            st.session_state.force_end = True
            st.session_state.end_reason = st.session_state.get("end_reason", "")
            st.rerun()
    with cols[2]:
        st.text_input("会話を終了したい理由", key="end_reason")

app()
