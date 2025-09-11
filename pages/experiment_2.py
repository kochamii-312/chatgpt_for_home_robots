import streamlit as st
import re
import json
import os
import joblib
from move_functions import move_to, pick_object, place_object_next_to, place_object_on
from openai import OpenAI
from dotenv import load_dotenv
from api import (
    client,
    SYSTEM_PROMPT_STANDARD,
    SYSTEM_PROMPT_FRIENDLY,
    SYSTEM_PROMPT_PRATFALL,
    build_bootstrap_user_message,
)
from jsonl import predict_with_model, save_experiment_2_result
from run_and_show import show_function_sequence, show_clarifying_question, run_plan_and_show
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
    model_path = st.session_state.get("model_path", "models/critic_model_20250903_053907.joblib")
    model = joblib.load(model_path)
    pred = model.predict(texts)
    return "sufficient" if pred[0] == 1 else "insufficient"

def app():
    st.title("LLMATCHデモアプリ")
    st.subheader("ChatGPT with 'Critic'")
    st.warning("このページは、再読み込み時にモデルの学習が行われるため、起動に時間がかかります。")
    
    st.sidebar.title("使用できる関数")
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
    YOLOで指定した物体を検出します。

    - **search_about(object:str)**  
    指定した物体に関する情報を検索します。

    - **push(object:str)**  
    指定した物体を押します。

    - **say(text:str)**  
    指定したテキストを発話します。
    """
    )

    prompt_options = {
        "Standard": SYSTEM_PROMPT_STANDARD,
        "Friendly": SYSTEM_PROMPT_FRIENDLY,
        "Pratfall": SYSTEM_PROMPT_PRATFALL,
    }
    prompt_label = st.selectbox("プロンプト", list(prompt_options.keys()))
    system_prompt = prompt_options[prompt_label]
    st.session_state["prompt_label"] = prompt_label

    model_files = [f for f in os.listdir("models") if f.endswith(".joblib")]
    if model_files:
        current_model = os.path.basename(st.session_state.get("model_path", model_files[0]))
        selected_model = st.selectbox(
            "評価モデル",
            model_files,
            index=model_files.index(current_model) if current_model in model_files else 0,
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
            "フォルダ",
            sub_options,
            index=sub_options.index(current_sub_label) if current_sub_label in sub_options else 0,
        )
        st.session_state["selected_subfolder"] = "" if sub_label == sub_default else sub_label
        if st.session_state["selected_subfolder"]:
            image_dir = os.path.join(image_dir, st.session_state["selected_subfolder"])
    else:
        st.session_state["selected_subfolder"] = ""

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
            "final_answer": "",
            "label": "",
            "clarifying_steps": []
        }
    if "active" not in st.session_state:
        st.session_state.active = True

    context = st.session_state["context"]

    message = st.chat_message("assistant")
    message.write("こんにちは、私は家庭用ロボットです！あなたの指示に従って行動します。")
    user_input = st.chat_input("ロボットへの指示や回答を入力してください")
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

        run_plan_and_show(reply)

    # sufficient判定なら終了
    label = get_critic_label(context)
    if label == "sufficient":
        st.success("クリティックモデルが「十分」と判定したため会話を終了します。")
        finalize_and_render_plan(label="sufficient")
        st.stop()

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
    
    if label == "sufficient":
        should_stop = True
        end_message = "モデルがsufficientを出力したため終了します。"

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
                save_experiment_2_result(scores)
                st.session_state.active = False
    
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
