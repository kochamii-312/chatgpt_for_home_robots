import streamlit as st
import streamlit.components.v1 as components
from consent import (
    apply_sidebar_hiding,
    configure_page,
    require_consent,
    should_hide_sidebar,
)
import json
import os
import random
import joblib

from dotenv import load_dotenv

from api import client, SYSTEM_PROMPT, build_bootstrap_user_message
from jsonl import predict_with_model, save_experiment_1_result
from move_functions_ import (
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
from image_task_sets import (
    build_task_set_choices,
    extract_task_lines,
    load_image_task_sets,
    resolve_image_paths,
)

load_dotenv()


configure_page(hide_sidebar_for_participant=True)


SCROLL_RESET_FLAG_KEY = "experiment1_scroll_reset_done"
ACTIVE_PAGE_STATE_KEY = "current_active_page"
ACTIVE_PAGE_VALUE = "experiment_1"


def _scroll_to_top_on_first_load() -> None:
    if st.session_state.get(ACTIVE_PAGE_STATE_KEY) != ACTIVE_PAGE_VALUE:
        st.session_state.pop(SCROLL_RESET_FLAG_KEY, None)

    if not st.session_state.get(SCROLL_RESET_FLAG_KEY):
        components.html(
            """
            <script>
            const doc = window.parent ? window.parent.document : document;
            const main = doc ? doc.querySelector('section.main') : null;
            if (main) {
                main.scrollTo(0, 0);
            } else {
                window.scrollTo(0, 0);
            }
            </script>
            """,
            height=0,
        )
        st.session_state[SCROLL_RESET_FLAG_KEY] = True

    st.session_state[ACTIVE_PAGE_STATE_KEY] = ACTIVE_PAGE_VALUE

def _reset_conversation_state(system_prompt: str) -> None:
    """Reset conversation-related session state for experiment 1."""

    st.session_state.context = [{"role": "system", "content": system_prompt}]
    st.session_state.active = True
    st.session_state.conv_log = {
        "label": "",
        "clarifying_steps": []
    }
    st.session_state.saved_jsonl = []
    st.session_state.turn_count = 0
    st.session_state.force_end = False
    # st.session_state.end_reason = []
    st.session_state["chat_input_history"] = []
    st.session_state["experiment1_followup_prompt"] = False
    st.session_state.pop("experiment1_followup_choice", None)
    _update_random_task_selection(
        "experiment1_selected_task_label",
        "experiment1_task_labels",
        "experiment1_label_to_key",
        "experiment1_selected_task_set",
    )


def _update_random_task_selection(label_key: str, labels_key: str, mapping_key: str, set_key: str) -> None:
    """Select a new task label at random and update related session state."""

    labels = st.session_state.get(labels_key) or []
    if not labels:
        return

    current_label = st.session_state.get(label_key)
    candidates = [label for label in labels if label != current_label] or labels
    new_label = random.choice(candidates)

    st.session_state[label_key] = new_label
    label_to_key = st.session_state.get(mapping_key) or {}
    st.session_state[set_key] = label_to_key.get(new_label)


def app():
    require_consent()
    _scroll_to_top_on_first_load()
    # st.title("LLMATCH Criticデモアプリ")
    st.markdown("### 実験1 GPTとGPT with Critic")

    if should_hide_sidebar():
        apply_sidebar_hiding()

    mode_options = ["GPT", "GPT with critic"]
    default_mode = st.session_state.get("mode", "GPT with critic")
    mode = st.radio("### ①モード選択", mode_options, index=mode_options.index(default_mode), horizontal=True)
    st.session_state["mode"] = mode

    st.write("※会話をリセットしてもこの選択は変わりません。")
    st.session_state.setdefault("critic_min_threshold", 0.60)
    
    system_prompt = SYSTEM_PROMPT
    
    with st.expander("評価モデル・タスク調整（任意）", expanded=False):
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
        st.session_state["critic_min_threshold"] = st.slider("critic_min_threshold", 0.5, 0.9, 0.60, 0.01)
        st.session_state["critic_margin"]       = st.slider("critic_margin", 0.0, 0.3, 0.15, 0.01)
        
        use_force = st.checkbox("しきい値を上書きする（force）", value=True)
        if use_force:
            st.session_state["critic_force_threshold"] = st.slider("force_threshold", 0.50, 0.90, 0.60, 0.01)
        else:
            st.session_state.pop("critic_force_threshold", None)

        task_sets = load_image_task_sets()
        if not task_sets:
            st.warning("写真とタスクのセットが保存されていません。まず『写真とタスクの選定・保存』ページで作成してください。")
            st.session_state["selected_image_paths"] = []
            st.session_state["experiment1_selected_task_set"] = None
            st.session_state["experiment1_task_labels"] = []
            st.session_state["experiment1_label_to_key"] = {}
        else:
            choice_pairs = build_task_set_choices(task_sets)
            labels = [label for label, _ in choice_pairs]
            label_to_key = {label: key for label, key in choice_pairs}

            st.session_state["experiment1_task_labels"] = labels
            st.session_state["experiment1_label_to_key"] = label_to_key

            if not labels:
                st.warning("保存済みのタスクが読み込めませんでした。")
                st.session_state["selected_image_paths"] = []
                st.session_state["experiment1_selected_task_set"] = None
                st.session_state["experiment1_task_labels"] = []
                st.session_state["experiment1_label_to_key"] = {}
                payload = {}
            else:
                stored_label = st.session_state.get("experiment1_selected_task_label")
                if stored_label not in labels:
                    stored_label = random.choice(labels)
                selected_label = st.selectbox(
                    "タスク",
                    labels,
                    index=labels.index(stored_label),
                )
                st.session_state["experiment1_selected_task_label"] = selected_label
                selected_task_name = label_to_key.get(selected_label)
                st.session_state["experiment1_selected_task_set"] = selected_task_name
                payload = task_sets.get(selected_task_name, {}) if selected_task_name else {}

        house = payload.get("house") if isinstance(payload, dict) else ""
        room = payload.get("room") if isinstance(payload, dict) else ""
        meta_lines = []

        task_lines = extract_task_lines(payload)

    st.markdown("#### ②指定されたタスク")
    st.write("下のタスクをそのまま画面下部のチャットに入力してください。")
    if task_lines:
        for line in task_lines:
            st.info(f"{line}")
    else:
        st.info("タスクが登録されていません。")

    image_candidates = []
    if isinstance(payload, dict):
        image_candidates = [str(p) for p in payload.get("images", []) if isinstance(p, str)]

    existing_images, missing_images = resolve_image_paths(image_candidates)

    st.session_state["selected_image_paths"] = existing_images

    if missing_images:
        st.warning(
            "以下の画像ファイルが見つかりません: " + ", ".join(missing_images)
        )

    st.markdown("#### ③指定されたタスクを行う場所")
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
        st.session_state.end_reason = []
    if "experiment1_followup_prompt" not in st.session_state:
        st.session_state["experiment1_followup_prompt"] = False

    st.markdown("#### ④ロボットとの会話")
    st.write("この下にロボットからの質問が表示されるので、③の写真を見ながら質問に対して答えてください。"
             "質問された情報が写真にない場合は、\"仮想の情報\"を答えて構いません。"
             "自動で評価フォームが表示されるまで会話を続けてください。")
    context = st.session_state["context"]

    message = st.chat_message("assistant")
    message.write("こんにちは、私は家庭用ロボットです！あなたの指示に従って行動します。")

    max_turns = 5
    should_stop = False
    label, p, th = None, None, None

    # 入力欄の表示制御
    if should_stop:
        user_input = None
    elif st.session_state.get("force_end"):
        user_input = None
    elif st.session_state.get("mode") == "GPT" and st.session_state.turn_count >= max_turns:
        st.info(f"{max_turns}回の操作に達したため、これ以上入力できません。（GPTモードのみ制限）")
        user_input = None
    else:
        input_box = st.chat_input("ロボットへの回答を入力してください", key="experiment_1_chat_input")
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
    assistant_messages = [m for m in context if m["role"] == "assistant"]
    if assistant_messages:
        label, p, th = predict_with_model()
        st.caption(f"評価モデルの予測: {label} (p={p:.3f}, th={th:.3f})")
    else:
        st.caption("評価モデルの予測: ---")

    last_assistant_content = assistant_messages[-1]["content"] if assistant_messages else ""
    has_plan = "<FunctionSequence>" in last_assistant_content
    high_conf = (p is not None and th is not None and p >= th + 0.15)

    should_stop = False
    end_message = ""
    if st.session_state.get("force_end"):
        should_stop = True
        end_message = "ユーザーが会話を終了しました。"
    elif st.session_state.get("mode") == "GPT with critic":
        if label == "sufficient" and (has_plan or high_conf or st.session_state.turn_count >= 2):
            should_stop = True
            end_message = "モデルがsufficientを出力したため終了します。"
    else:
        if st.session_state.turn_count >= 4:
            should_stop = True
            end_message = "4回の会話に達したため終了します。"

    if should_stop:
        if st.session_state.active == True:
            st.success(end_message)
            with st.form("evaluation_form"):
                st.subheader("⑤評価フォーム")
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
                        "成功する",
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
                        "コミュニケーションが明確でなかった（何と答えればいいかわからない質問があった等）",
                        "特になし",
                    ]
                )
                kindness = st.radio(
                    "ロボットはあなたに対してどれくらい「親切さ/丁寧さ」を持って接していましたか？",
                    ["非常に親切だった", "まあまあ親切だった", "どちらともいえない", "あまり親切でなかった", "全く親切でなかった"],
                    horizontal=True
                )
                pleasantness = st.radio(
                    "ロボットとの会話はどれくらい「愉快さ」を感じましたか？",
                    ["非常に愉快だった", "まあまあ愉快だった", "どちらともいえない", "少し不愉快だった", "とても不愉快だった"],
                    horizontal=True
                )
                familiarity = st.radio(
                    "ロボットにどれくらい「親近感/親しみやすさ（=心理的距離感の近さ）」を持ちましたか？",
                    ["強く持った", "まあまあ持った", "どちらともいえない", "あまり持ってない", "全く持っていない"],
                    horizontal=True
                )
                social_presence = st.radio(
                    "対話の相手がそこに存在し、自分と同じ空間を共有している、あるいは自分と関わっている感覚「ソーシャルプレゼンス（=存在感）」をどれくらい持ちましたか？",
                    ["強く持った", "まあまあ持った", "どちらともいえない", "あまり持ってない", "全く持っていない"],
                    horizontal=True
                )
                security = st.radio(
                    "ロボットに対してどれくらい「安心感/信頼感」を持ちましたか？",
                    ["強く持った", "まあまあ持った", "どちらともいえない", "あまり持ってない", "全く持っていない"],
                    horizontal=True
                )
                impression = st.text_input(
                    "AIとの会話や、ロボットの行動計画について「印象に残ったこと」があればお願いします。"
                )
                free = st.text_input(
                    "その他に何か感じたことがあればお願いします。"
                )
                submitted = st.form_submit_button("評価を保存")

            if submitted:
                st.warning("評価を保存しました！適宜休憩をとってください☕")
                scores = {
                    "name": name,
                    "success": success,
                    "failure_reason": failure_reason,
                    "failure_reason_others": failure_reason_others,
                    "grices_maxim": grices_maxim,
                    "kindness": kindness,
                    "pleasantness": pleasantness,
                    "familiarity": familiarity,
                    "social_presence": social_presence,
                    "security": security,
                    "impression": impression,
                    "free": free,
                }
                termination_label = "会話を強制的に終了" if st.session_state.get("force_end") else ""
                selected_reasons = st.session_state.get("end_reason", [])
                if isinstance(selected_reasons, str):
                    termination_reason = selected_reasons
                else:
                    termination_reason = "、".join(selected_reasons)
                save_experiment_1_result(
                    scores,
                    termination_reason,
                    termination_label,
                )
                st.session_state.active = False
                st.session_state["experiment1_followup_prompt"] = True
                st.session_state.pop("experiment1_followup_choice", None)   

    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("⚠️会話をリセット", key="reset_conv"):
            _reset_conversation_state(system_prompt)
            st.rerun()
    with cols[1]:
        if st.button("🚨会話を終了", key="force_end_button"):
            st.session_state.force_end = True
            st.session_state.end_reason = st.session_state.get("end_reason", [])
            st.rerun()
    with cols[2]:
        st.multiselect(
            "会話を終了したい理由",
            [
                "行動計画は実行可能でさらなる質問は不要",
                "同じ質問が繰り返される",
                "計画が確定している",
                "LLMから質問されない",
                "その他",
            ],
            key="end_reason",
        )
    if st.session_state.get("experiment1_followup_prompt"):
        st.markdown("**GPTモード** と **GPT with Criticモード** で1回ずつ実験を終えましたか？")
        if st.button("🙅‍♂️いいえ → ①のモードを変えて再度実験", key="followup_no", type="primary"):
            st.session_state["experiment1_followup_prompt"] = False
            st.session_state.pop("experiment1_followup_choice", None)
            _reset_conversation_state(system_prompt)
            st.rerun()
        if st.button("🙆‍♂️はい → 実験2", key="followup_yes", type="primary"):
            st.session_state["experiment1_followup_prompt"] = False
            st.session_state.pop("experiment1_followup_choice", None)
            st.session_state.pop(SCROLL_RESET_FLAG_KEY, None)
            st.session_state.pop("experiment2_scroll_reset_done", None)
            st.switch_page("pages/experiment_2.py")

app()
