import json
import os
import random
import re

import joblib
import streamlit as st
from dotenv import load_dotenv

from api import (
    SYSTEM_PROMPT_FRIENDLY,
    SYSTEM_PROMPT_PRATFALL,
    SYSTEM_PROMPT_STANDARD,
    build_bootstrap_user_message,
    client,
)
from jsonl import predict_with_model, save_experiment_2_result
from move_functions import move_to, pick_object, place_object_next_to, place_object_on
from run_and_show import run_plan_and_show, show_clarifying_question, show_function_sequence
from image_task_sets import (
    build_task_set_choices,
    extract_task_lines,
    load_image_task_sets,
    resolve_image_paths,
)

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
    st.title("LLMATCH Criticデモアプリ")
    st.subheader("実験2 異なるコミュニケーションタイプの比較")
    
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

    prompt_options = {
        "1": SYSTEM_PROMPT_STANDARD,
        "2": SYSTEM_PROMPT_FRIENDLY,
        "3": SYSTEM_PROMPT_PRATFALL,
    }
    prompt_keys = list(prompt_options.keys())
    if "prompt_label" not in st.session_state:
        st.session_state["prompt_label"] = random.choice(prompt_keys)

    default_prompt_label = st.session_state["prompt_label"]
    prompt_label = st.selectbox(
        "プロンプト（自動）",
        prompt_keys,
        index=prompt_keys.index(default_prompt_label)
        if default_prompt_label in prompt_keys
        else 0,
    )
    system_prompt = prompt_options[prompt_label]
    st.session_state["prompt_label"] = prompt_label

    with st.expander("評価モデル・タスク調整（任意）", expanded=False):
        # 評価モデル selectbox
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
                "評価モデル（自動）",
                model_files,
                index=model_files.index(current_model),
            )
            st.session_state["model_path"] = os.path.join("models", selected_model)

        # タスク selectbox
        task_sets = load_image_task_sets()
        if not task_sets:
            st.warning("写真とタスクのセットが保存されていません。まず『写真とタスクの選定・保存』ページで作成してください。")
            st.session_state["selected_image_paths"] = []
            st.session_state["experiment2_selected_task_set"] = None
        else:
            choice_pairs = build_task_set_choices(task_sets)
            labels = [label for label, _ in choice_pairs]
            label_to_key = {label: key for label, key in choice_pairs}

            if not labels:
                st.warning("保存済みのタスクが読み込めませんでした。")
                st.session_state["selected_image_paths"] = []
                st.session_state["experiment2_selected_task_set"] = None
                payload = {}
            else:
                # ランダム選択
                selected_label = random.choice(labels)
                st.selectbox("タスク", labels, index=labels.index(selected_label))
                selected_task_name = label_to_key.get(selected_label)
                st.session_state["experiment2_selected_task_set"] = selected_task_name
                payload = task_sets.get(selected_task_name, {}) if selected_task_name else {}
        house = payload.get("house") if isinstance(payload, dict) else ""
        room = payload.get("room") if isinstance(payload, dict) else ""
        meta_lines = []

        task_lines = extract_task_lines(payload)

    st.markdown("### 指定されたタスク")
    if task_lines:
        for line in task_lines:
            st.info(f"{line}")
    else:
        st.info("タスクが登録されていません。")
    st.write("→指定されたタスクをそのままテキストフィールドに入力してください！")

    image_candidates = []
    if isinstance(payload, dict):
        image_candidates = [str(p) for p in payload.get("images", []) if isinstance(p, str)]

    existing_images, missing_images = resolve_image_paths(image_candidates)

    st.session_state["selected_image_paths"] = existing_images

    if missing_images:
        st.warning(
            "以下の画像ファイルが見つかりません: " + ", ".join(missing_images)
        )

    st.markdown("### 指定されたタスクが行われる場所")
    if house:
        meta_lines.append(f"家: {house}")
    if room:
        meta_lines.append(f"部屋: {room}")
    if meta_lines:
        st.write(" / ".join(meta_lines))
    if existing_images:
        for path in existing_images:
            st.image(path, caption=os.path.basename(path))
    else:
        st.info("画像が設定されていません。")

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
        st.session_state["chat_input_history"] = []
    if "active" not in st.session_state:
        st.session_state.active = True
    if "force_end" not in st.session_state:
        st.session_state.force_end = False
    if "end_reason" not in st.session_state:
        st.session_state.end_reason = ""
    if "chat_input_history" not in st.session_state:
        st.session_state["chat_input_history"] = []

    st.markdown("### ロボットとの会話")
    st.write("最初にタスクを入力し、上の写真を見ながらロボットの質問に対して答えてください。")
    context = st.session_state["context"]

    message = st.chat_message("assistant")
    message.write("こんにちは、私は家庭用ロボットです！あなたの指示に従って行動します。")
    if st.session_state.get("force_end"):
        user_input = None
    else:
        user_input = st.chat_input("ロボットへの指示や回答を入力してください")
        if user_input:
            st.session_state["chat_input_history"].append(user_input)
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

    if st.session_state.get("force_end"):
        should_stop = True
        end_message = "ユーザーが会話を強制的に終了しました。"
    elif label == "sufficient":
        should_stop = True
        end_message = "モデルがsufficientを出力したため終了します。"

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
                save_experiment_2_result(
                    scores,
                    st.session_state.get("end_reason", ""),
                    termination_label,
                )
                st.session_state.active = False

    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("⚠️会話をリセット", key="reset_conv"):
            # セッション情報を初期化
            st.session_state.context = [{"role": "system", "content": system_prompt}]
            st.session_state.active = True
            st.session_state.conv_log = {
                "label": "",
                "clarifying_steps": []
            }
            st.session_state.saved_jsonl = []
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
