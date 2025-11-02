import json
import random
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import yaml
from consent import (
    apply_sidebar_hiding,
    configure_page,
    require_consent,
    should_hide_sidebar,
)
from dotenv import load_dotenv

from api import build_bootstrap_user_message, client
from jsonl import (
    save_conversation_history_to_firestore,
    save_experiment_2_result,
)
from move_functions import move_to, pick_object, place_object_next_to, place_object_on
from run_and_show import run_plan_and_show, show_spoken_response, show_function_sequence
from image_task_sets import (
    build_task_set_choices,
    extract_task_lines,
    load_image_task_sets,
    resolve_image_paths,
)
from two_classify import prepare_data  # 既存関数を利用
from esm import ExternalStateManager

PROMPT_GROUP = "smalltalk"
NEXT_PAGE = None

PROMPT_TASKINFO_PATH = Path(__file__).resolve().parent.parent / "json" / "prompt_taskinfo_sets.yaml"
_PROMPT_TASKINFO_CACHE: dict[str, dict[str, str]] | None = None


def load_prompt_taskinfo_sets() -> dict[str, dict[str, str]]:
    global _PROMPT_TASKINFO_CACHE
    if _PROMPT_TASKINFO_CACHE is None:
        with PROMPT_TASKINFO_PATH.open(encoding="utf-8") as f:
            _PROMPT_TASKINFO_CACHE = yaml.safe_load(f)
    return _PROMPT_TASKINFO_CACHE


def get_prompt_options(prompt_group: str) -> dict[str, dict[str, str]]:
    return {
        key: value
        for key, value in load_prompt_taskinfo_sets().items()
        if value.get("prompt_group") == prompt_group
    }

SUS_OPTIONS = [
    ("とても当てはまる (5)", 5),
    ("やや当てはまる (4)", 4),
    ("どちらでもない (3)", 3),
    ("あまり当てはまらない (2)", 2),
    ("まったく当てはまらない (1)", 1),
]

SUS_QUESTIONS = [
    ("sus_q1", "このロボットを頻繁に使用したい"),
    ("sus_q2", "このロボットは必要以上に複雑だと思う"),
    ("sus_q3", "このロボットは使いやすいと感じた"),
    ("sus_q4", "このロボットを使うには専門的なサポートが必要だ"),
    ("sus_q5", "このロボットの様々な機能は統合されていると感じた"),
    ("sus_q6", "このロボットは一貫性が欠けていると思う"),
    ("sus_q7", "大半の人はこのロボットをすぐに使いこなせるようになると思う"),
    ("sus_q8", "このロボットは操作しにくい"),
    ("sus_q9", "このロボットを使いこなせる自信がある"),
    ("sus_q10", "このロボットを使い始める前に知らなければならないことがたくさんあると思う"),
]

NASA_TLX_QUESTIONS = [
    ("nasa_mental_demand", "あなたは、ロボットと会話をするにあたって、精神的要求（思考，意志決定，計算，記憶，観察，検索，等）がどれくらい要求されましたか？"),
    ("nasa_physical_demand", "あなたは、ロボットと会話をするにあたって、身体的要求（押す，引く，回す， 操作する等）がどれくらい要求されましたか？"),
    ("nasa_temporal_demand", "あなたは、ロボットと会話をするにあたって、時間的切迫感（作業や要素作業の頻度や速さ）をどの程度感じましたか？"),
    ("nasa_performance", "ロボットと会話をするにあたって、あなた自身が設定した作業（指示）は、どの程度ロボットによって達成されたと考えますか？"),
    ("nasa_effort", "あなたはその作業達成率に到達するのに、どのくらい（精神的および身体的に）努力しましたか？"),
    ("nasa_frustration", "あなたは、ロボットと会話をするにあたってどのくらい不安，落胆，いらいら，ストレス，不快感を感じましたか？"),
]


GodSpeed_anthroporphism_QUESTIONS = [
    ("godspeed_anthroporphism1", "Fake 偽物のような (1) - Natural 自然な (5)"),
    ("godspeed_anthroporphism2", "Machinelike 機械的 (1) - Humanlike 人間的 (5)"),
    ("godspeed_anthroporphism3", "Unconscious 意識を持たない (1) - Contious 意識を持っている (5)"),
    ("godspeed_anthroporphism4", "Artificial 人工的 (1) - Lifelike 生物的 (5)"),
    ("godspeed_anthroporphism5", "Moving rigidly ぎこちない動き (1) - Moving elegantly 洗練された動き (1)")
]

GodSpeed_animacy_QUESTIONS = [
    ("godspeed_animacy1", "Dead 死んでいる (1) - Alive 生きている (5)"),
    ("godspeed_animacy2", "Stagnant 活気のない (1) - Lively 生き生きとした (5)"),
    ("godspeed_animacy3", "Mechanical 機械的な (1) - Organic 有機的な (5)"),
    ("godspeed_animacy4", "Inert 不活発な (1) - Interactive 対話的な (5)"),
    ("godspeed_animacy5", "Apathetic 無関心な (1) - Responsive 反応のある (5)")
]

GodSpeed_likebility_QUESTIONS = [
    ("godspeed_likeability1", "Dislike 嫌い (1) - Like 好き (5)"),
    ("godspeed_likeability2", "Unfriendly 親しみにくい (1) - Friendly 親しみやすい (5)"),
    ("godspeed_likeability3", "Unkind 不親切な (1) - Kind 親切な (5)"),
    ("godspeed_likeability4", "Unpleasant 不愉快な (1) - Pleasant 愉快な (5)"),
    ("godspeed_likeability5", "Awful ひどい (1) - Nice 良い (5)")
]

GodSpeed_perceived_intelligence_QUESTIONS = [
    ("godspeed_intelligence1", "Incompetent 無能な (1) - Competent 有能な (5)"),
    ("godspeed_intelligence2", "Ignorant 無知な (1) - Knowledgeable 物知りな (5)"),
    ("godspeed_intelligence3", "Irresponsible 無責任な (1) - Responsible 責任のある (5)"),
    ("godspeed_intelligence4", "Unintelligent 知的でない (1) - Intelligent 知的な (5)"),
    ("godspeed_intelligence5", "Foolish 愚かな (1) - Sensible 賢明な (5)")
]

GodSpeed_perceived_safety_QUESTIONS = [
    ("godspeed_safety1", "Anxious 不安な (1) - Relaxed 落ち着いた (5)"),
    ("godspeed_safety2", "Agitated 動揺している (1) - Calm 冷静な (5)"),
    ("godspeed_safety3", "Quiescent 平穏な (1) - Surprised 驚いた (5)")
]

load_dotenv()


configure_page(hide_sidebar_for_participant=True)


def _reset_conversation_state(system_prompt: str) -> None:
    """Reset conversation-related session state for experiment 2."""

    # 1. ESM（状態）の初期化
    st.session_state.esm = ExternalStateManager() 
    
    # 2. 実行すべき行動計画のキュー（名前を action_plan_queue に統一）
    st.session_state.action_plan_queue = [] 
    
    # 3. フェーズ1（目標設定）が完了したかのフラグ
    st.session_state.goal_set = False 
    
    # 4. システムプロンプトを「テンプレート」として保持
    #    (LLM呼び出しの度に {current_state_xml} を埋め込むため)
    st.session_state.system_prompt_template = system_prompt 
    
    # 5. contextは「空」で開始する
    st.session_state.context = [] 
    
    # --- 以下は既存のリセットロジック ---
    st.session_state.active = True
    st.session_state.conv_log = {
        "label": "",
        "clarifying_steps": []
    }
    st.session_state.saved_jsonl = []
    st.session_state.turn_count = 0
    st.session_state.force_end = False
    st.session_state["chat_input_history"] = []
    st.session_state["experiment2_followup_prompt"] = False
    st.session_state.pop("experiment2_followup_choice", None)
    _update_random_task_selection(
        "experiment2_selected_task_label",
        "experiment2_task_labels",
        "experiment2_label_to_key",
        "experiment2_selected_task_set",
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

TAG_RE = re.compile(r"</?([A-Za-z0-9_]+)(\s[^>]*)?>")

def strip_tags(text: str) -> str:
    return TAG_RE.sub("", text or "").strip()

def extract_between(tag: str, text: str) -> str | None:
    match = re.search(fr"<{tag}>([\s\S]*?)</{tag}>", text or "", re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_xml_tag(xml_string, tag_name):
    """指定されたタグの内容を抽出する"""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, xml_string, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def parse_function_sequence(sequence_str):
    """FunctionSequenceの番号付きリストをパースする"""
    if not sequence_str:
        return []
    # "1. go to..." "2. pick up..." などを抽出
    actions = re.findall(r'^\s*\d+\.\s*(.*)', sequence_str, re.MULTILINE)
    return [action.strip() for action in actions]

def safe_format_prompt(template: str, **kwargs) -> str:
    # {current_state_xml},{house},{room} だけを置換し、他の { ... } は触らない
    pattern = re.compile(r"\{(current_state_xml|house|room)\}")
    return pattern.sub(lambda m: str(kwargs.get(m.group(1), m.group(0))), template)

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

def _build_text_for_model(instruction: str, function_sequence: str, information: str) -> str:
    # 学習時(two_classify.py)の prepare_data と同じ接頭辞・結合順に合わせる
    parts = []
    if instruction.strip():
        parts.append(f"Instruction: {instruction.strip()}")
    if function_sequence.strip():
        parts.append(f"FunctionSequence: {function_sequence.strip()}")
    if information.strip():
        parts.append(f"Information: {information.strip()}")
    return " | ".join(parts)

def _extract_between(tag: str, text: str) -> str | None:
    m = re.search(fr"<{tag}>([\s\S]*?)</{tag}>", text or "", re.IGNORECASE)
    return m.group(1).strip() if m else ""

def app():
    # require_consent()
    st.markdown("### 実験 異なるコミュニケーションタイプの比較")

    if should_hide_sidebar():
        apply_sidebar_hiding()

    prompt_options = get_prompt_options(PROMPT_GROUP)
    if not prompt_options:
        st.error("指定されたプロンプトグループに対応するプロンプトが見つかりませんでした。")
        return

    prompt_keys = list(prompt_options.keys())
    prompt_label_state_key = f"experiment2_{PROMPT_GROUP}_prompt_label"
    if prompt_label_state_key not in st.session_state:
        st.session_state[prompt_label_state_key] = random.choice(prompt_keys)

    default_prompt_label = st.session_state[prompt_label_state_key]
    st.markdown("#### ①プロンプト選択（自動）")
    prompt_label = st.selectbox(
        "選択肢",
        prompt_keys,
        index=prompt_keys.index(default_prompt_label)
        if default_prompt_label in prompt_keys
        else 0,
    )
    selected_prompt = prompt_options[prompt_label]
    system_prompt = selected_prompt.get("prompt", "")
    selected_task_name = selected_prompt.get("task", "")
    selected_taskinfo = selected_prompt.get("taskinfo", "")

    if not system_prompt:
        st.error("プロンプトの内容が設定されていません。JSONファイルを確認してください。")
        return

    st.session_state[prompt_label_state_key] = prompt_label

    st.write("※会話をリセットしてもこの選択は変わりません。")

    payload = None
    with st.expander("タスク調整（任意）", expanded=False):
        task_sets = load_image_task_sets()
        if not task_sets:
            st.warning("写真とタスクのセットが保存されていません。まず『写真とタスクの選定・保存』ページで作成してください。")
            st.session_state["selected_image_paths"] = []
            st.session_state["experiment2_selected_task_set"] = None
            st.session_state["experiment2_task_labels"] = []
            st.session_state["experiment2_label_to_key"] = {}
            payload = {}
        else:
            choice_pairs = build_task_set_choices(task_sets)
            labels = [label for label, _ in choice_pairs]
            label_to_key = {label: key for label, key in choice_pairs}

            st.session_state["experiment2_task_labels"] = labels
            st.session_state["experiment2_label_to_key"] = label_to_key

            if not labels:
                st.warning("保存済みのタスクが読み込めませんでした。")
                st.session_state["selected_image_paths"] = []
                st.session_state["experiment2_selected_task_set"] = None
                st.session_state["experiment2_task_labels"] = []
                st.session_state["experiment2_label_to_key"] = {}
                payload = {}
            else:
                stored_label = st.session_state.get("experiment2_selected_task_label")
                if stored_label not in labels:
                    stored_label = random.choice(labels)
                selected_label = st.selectbox(
                    "タスク",
                    labels,
                    index=labels.index(stored_label),
                )
                st.session_state["experiment2_selected_task_label"] = selected_label
                selected_task_name = label_to_key.get(selected_label)
                st.session_state["experiment2_selected_task_set"] = selected_task_name
                payload = task_sets.get(selected_task_name, {}) if selected_task_name else {}
    if not isinstance(payload, dict):
        payload = {}

    house = payload.get("house") if isinstance(payload, dict) else ""
    room = payload.get("room") if isinstance(payload, dict) else ""
    meta_lines = []

    task_lines = extract_task_lines(payload)

    st.markdown("#### ②指定されたタスク")
    st.write("下のタスクをそのまま画面下部のチャットに入力してください。")
    if selected_taskinfo:
        st.info(selected_taskinfo)
    else:
        st.info("タスクが登録されていません。")
    # if task_lines:
    #     for line in task_lines:
    #         st.info(f"{line}")
    # else:
    #     st.info("タスクが登録されていません。")

    # 1) セッションにESMとコンテキストを初期化
    if (
        "esm" not in st.session_state
        or st.session_state.get("system_prompt_template") != system_prompt
    ):
        _reset_conversation_state(system_prompt) 

    # セッションからESMオブジェクトを取得
    esm = st.session_state.esm
    
    if "active" not in st.session_state:
        st.session_state.active = True
    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0
    if "force_end" not in st.session_state:
        st.session_state.force_end = False
    if "chat_input_history" not in st.session_state:
        st.session_state["chat_input_history"] = []
    if "experiment2_followup_prompt" not in st.session_state:
        st.session_state["experiment2_followup_prompt"] = False

    context = st.session_state.context
    esm = st.session_state.esm
    queue = st.session_state.action_plan_queue
    current_state = esm.current_state
    should_stop = False
    end_message = ""

    tab_conversation, tab_state = st.tabs([
        "④ロボットとの会話",
        "③現在の状態",
    ])

    with tab_conversation:
        st.markdown("#### ④ロボットとの会話")
        st.write(
            "最初に②のタスクを入力し、ロボットと自由に会話してください。"
            "最終的にはロボットと一緒に、タスクを達成させてください。"
        )

        # 2. 既存の会話履歴を表示
        for msg in context:
            if msg["role"] == "system":
                continue
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                # 既存のヘルパー関数をそのまま利用
                if msg["role"] == "assistant":
                    reply_xml = msg.get("full_reply", msg.get("content", ""))
                    show_function_sequence(reply_xml)
                    # show_spoken_response(reply_xml)

        # 3. [フェーズ2: 実行ループ] 実行すべき行動計画（キュー）があるか？
        if queue:
            next_action = queue[0]
            st.info(f"次の行動計画: **{next_action}**")

            # 実行ボタン
            if st.button(f"▶️ 実行: {next_action}", key="run_next_step", type="primary"):
                action_to_run = queue.pop(0)  # キューの先頭を取り出す
                st.session_state.action_plan_queue = queue  # キューを更新

                # [!!!] ここで実際のロボットAPIを呼び出す（代わりにESMを更新）[!!!]
                with st.spinner(f"実行中: {action_to_run}..."):
                    # time.sleep(1) # import time が必要
                    esm.update_state_from_action(action_to_run)

                # 実行結果を会話履歴（コンテキスト）に追加
                exec_msg = (
                    f"（実行完了: {action_to_run}。ロボットの状態を更新しました。）"
                )
                context.append({"role": "user", "content": exec_msg})  # 実行結果をLLMに伝える
                st.chat_message("user").write(exec_msg)

                # キューが空になったら、LLMに次の計画を尋ねる
                if not queue:
                    st.info("サブタスクが完了しました。LLMに次の計画を問い合わせます...")
                    context.append(
                        {
                            "role": "user",
                            "content": "このサブタスクは完了しました。現在の状態に基づき、次のサブタスクを計画してください。",
                        }
                    )
                    st.session_state.trigger_llm_call = True

                st.rerun()  # 画面を再描画して次のステップを表示

        # 4. LLM呼び出しのトリガー（ユーザー入力 or 計画完了）
        user_input = None
        if not st.session_state.get("force_end"):
            user_input = st.chat_input(
                "ロボットへの回答を入力してください",
                key="experiment_2_chat_input",
            )
            if user_input:
                st.session_state["chat_input_history"].append(user_input)
                st.session_state.trigger_llm_call = True

                # ユーザーが入力した=既存の計画に介入した→したがって古い行動計画（キュー）を破棄する
                if queue:
                    st.warning("ユーザーが介入しました。既存の行動計画を破棄します。")
                    st.session_state.action_plan_queue = []
                    queue = []

        # 5. [フェーズ1 & 2: LLM呼び出し]
        if st.session_state.get("trigger_llm_call"):
            st.session_state.trigger_llm_call = False  # フラグをリセット

            # [変更点] ユーザー入力があった場合のみコンテキストに追加
            if user_input:
                context.append({"role": "user", "content": user_input})

            # [!!!] LLM呼び出しのコアロジック [!!!]
            with st.chat_message("assistant"):
                with st.spinner("ロボットが考えています..."):
                    # (A) ESMから最新の状態XMLを取得
                    current_state_xml = esm.get_state_as_xml_prompt()
                    # (B) 最新の状態でシステムプロンプトを構築
                    house = (payload.get("house") if isinstance(payload, dict) else "") or ""
                    room = (payload.get("room") if isinstance(payload, dict) else "") or ""
                    system_prompt_content = safe_format_prompt(
                        st.session_state.system_prompt_template,
                        current_state_xml=current_state_xml,
                        house=house,
                        room=room,
                    )
                    system_message = {"role": "system", "content": system_prompt_content}

                    # (C) APIに渡すメッセージリストを作成
                    messages_for_api = [system_message] + context

                    # (D) LLM API 呼び出し
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",  # または "gpt-4-turbo"
                        messages=messages_for_api,
                    )
                    reply = response.choices[0].message.content.strip()

                    # (E) 応答をコンテキストに追加
                    spoken_response = extract_xml_tag(reply, "SpokenResponse")
                    if not spoken_response:
                        spoken_response = strip_tags(reply) or "(...)"

                    context.append(
                        {
                            "role": "assistant",
                            "content": spoken_response,
                            "full_reply": reply,
                        }
                    )
                    st.session_state.turn_count += 1

                    # (F) [フェーズ1] Goalが設定されたかパース
                    goal_def_str = extract_xml_tag(reply, "TaskGoalDefinition")
                    if (
                        goal_def_str
                        and "Goal:" in goal_def_str
                        and not st.session_state.goal_set
                    ):
                        if esm.set_task_goal_from_llm(goal_def_str):
                            st.session_state.goal_set = True
                            st.success("タスク目標を設定しました！")
                        else:
                            st.error("LLMが生成したタスク目標のパースに失敗しました。")

                    # (G) [フェーズ2] 行動計画が生成されたかパース
                    plan_str = extract_xml_tag(reply, "FunctionSequence")
                    if plan_str:
                        # [変更点] 介入時に古い計画がクリアされているため、extendでOK
                        actions = parse_function_sequence(plan_str)
                        if actions:
                            st.session_state.action_plan_queue.extend(actions)
                            st.info(f"{len(actions)}ステップの計画を受信しました。")

                    # (H) 画面を再描画
                    st.rerun()

    if st.session_state.get("force_end"):
        should_stop = True
        end_message = "ユーザーが会話を終了しました。"

    with tab_state:
        st.markdown("#### ③現在の状態")
        st.caption(
            "ExternalStateManager (ESM) が保持している状態です。ロボットの行動に応じて更新されます。"
        )

        # --- 1. ロボットの状態 ---
        st.markdown("##### 🤖 ロボット")
        col1, col2 = st.columns(2)

        # esm.py のキーに合わせて指定
        robot_stat = current_state.get("robot_status", {})
        location = robot_stat.get("location", "不明")
        holding = robot_stat.get("holding", "なし")

        # 'living_room' -> 'Living Room' のように整形して表示
        col1.metric("現在地", location.replace("_", " ").title())
        col2.metric("掴んでいる物", str(holding) if holding else "なし")

        st.divider()  # 区切り線

        # --- 2. 環境の状態 ---
        st.markdown("##### 🏠 環境（場所ごとのアイテム）")
        environment_state = current_state.get("environment", {})

        # 場所が多いため2列に分けて表示
        env_cols = st.columns(2)

        # 辞書のキー（場所）を半分に分ける
        locations = list(environment_state.keys())
        mid_point = (len(locations) + 1) // 2
        locations_col1 = locations[:mid_point]
        locations_col2 = locations[mid_point:]

        # 左側の列
        with env_cols[0]:
            for loc in locations_col1:
                items = environment_state.get(loc, [])
                # 'kitchen_shelf' -> 'Kitchen Shelf'
                loc_label = loc.replace("_", " ").title()

                with st.expander(f"{loc_label} ({len(items)}個)"):
                    if items:
                        st.multiselect(
                            f"（{loc_label}にある物）",
                            items,
                            default=items,
                            disabled=True,
                            label_visibility="collapsed",  # ラベルを非表示に
                        )
                    else:
                        st.info("（何もありません）")

        # 右側の列
        with env_cols[1]:
            for loc in locations_col2:
                items = environment_state.get(loc, [])
                loc_label = loc.replace("_", " ").title()

                with st.expander(f"{loc_label} ({len(items)}個)"):
                    if items:
                        st.multiselect(
                            f"（{loc_label}にある物）",
                            items,
                            default=items,
                            disabled=True,
                            label_visibility="collapsed",
                        )
                    else:
                        st.info("（何もありません）")

        # --- 3. タスク目標 (ついでに表示) ---
        st.divider()
        st.markdown("##### 🎯 現在のタスク目標")
        task_goal = current_state.get("task_goal", {})
        target_loc = task_goal.get("target_location", "未設定")
        items_needed = task_goal.get("items_needed", {})

        col_t1, col_t2 = st.columns(2)
        col_t1.metric("目標地点", str(target_loc).title() if target_loc else "未設定")

        if items_needed:
            # 辞書 { 'itemA': 2, 'itemB': 1 } をリスト表示
            item_list = [f"{item} (x{count})" for item, count in items_needed.items()]
            col_t2.markdown("**必要なアイテム:**")
            col_t2.dataframe(
                item_list,
                use_container_width=True,
                hide_index=True,
                column_config={"value": "アイテム (個数)"},
            )
        else:
            col_t2.metric("必要なアイテム", "なし")

        # --- 元のJSONはデバッグ用に折りたたんで残す ---
        with st.expander("詳細な状態（JSON）"):
            st.json(current_state)

    # 7. 評価フォームの表示（should_stop判定ロジックは変更済み）  
    end_message = ""
    if st.session_state.get("force_end"):
        should_stop = True
        end_message = "ユーザーが会話を終了しました。"
    else:
        pass

    if should_stop:
        if st.session_state.active == True:
            st.success(end_message)
            with st.form("evaluation_form"):
                st.subheader("⑥評価フォーム")
                name = st.text_input(
                    "あなたの名前やユーザーネーム等（被験者区別用）"
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

                st.markdown("###### SUS（システムユーザビリティ尺度）")
                sus_scores = {}
                sus_option_labels = [label for label, _ in SUS_OPTIONS]
                sus_value_map = dict(SUS_OPTIONS)
                for key, question in SUS_QUESTIONS:
                    choice = st.radio(
                        question,
                        sus_option_labels,
                        horizontal=True,
                        key=f"{key}_experiment2",
                    )
                    sus_scores[key] = sus_value_map.get(choice)

                st.markdown("###### NASA TLX（1 = 低い ／ 5 = 高い）")
                nasa_scores = {}
                for key, question in NASA_TLX_QUESTIONS:
                    nasa_scores[key] = st.slider(
                        question,
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1,
                        format="%d",
                        key=f"{key}_experiment2",
                    )

                st.markdown("###### Godspeed ロボットの印象について")
                st.markdown("**・人間らしさ（Anthropomorphism）**: 以下のスケールに基づいてこのロボットの印象を評価してください。")
                godspeed_anthroporphism_scores = {}
                for key, question in GodSpeed_anthroporphism_QUESTIONS:
                    godspeed_anthroporphism_scores[key] = st.slider(
                        question,
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1,
                        format="%d",
                        key=f"{key}_experiment2",
                    )
                st.markdown("**・生命感（Animacy）**: 以下のスケールに基づいてこのロボットの印象を評価してください。")
                godspeed_animacy_scores = {}
                for key, question in GodSpeed_animacy_QUESTIONS:
                    godspeed_animacy_scores[key] = st.slider(
                        question,
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1,
                        format="%d",
                        key=f"{key}_experiment2",
                    )
                st.markdown("**・好感度（Likeability）**: 以下のスケールに基づいてこのロボットの印象を評価してください。")
                godspeed_likeability_scores = {}
                for key, question in GodSpeed_likebility_QUESTIONS:
                    godspeed_likeability_scores[key] = st.slider(
                        question,
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1,
                        format="%d",
                        key=f"{key}_experiment2",
                    )
                st.markdown("**・知能の知覚（Perceived Intelligence）**: 以下のスケールに基づいてあなたの心の状態を評価してください。")
                godspeed_intelligence_scores = {}
                for key, question in GodSpeed_perceived_intelligence_QUESTIONS:
                    godspeed_intelligence_scores[key] = st.slider(
                        question,
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1,
                        format="%d",
                        key=f"{key}_experiment2",
                    )
                st.markdown("**・安全性の知覚（Perceived Safety）**")
                godspeed_safety_scores = {}
                for key, question in GodSpeed_perceived_safety_QUESTIONS:
                    godspeed_safety_scores[key] = st.slider(
                        question,
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1,
                        format="%d",
                        key=f"{key}_experiment2",
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
                    "impression": impression,
                    "free": free,
                }
                # scores.update(sus_scores)
                scores.update(nasa_scores)
                scores.update(godspeed_anthroporphism_scores)
                scores.update(godspeed_animacy_scores)
                scores.update(godspeed_likeability_scores)
                scores.update(godspeed_intelligence_scores)
                scores.update(godspeed_safety_scores)
                termination_label = (
                    "タスク完了ボタンが押されました"
                    if st.session_state.get("force_end")
                    else ""
                )
                save_experiment_2_result(
                    scores,
                    termination_label=termination_label,
                )
                st.session_state.active = False
                st.session_state["experiment2_followup_prompt"] = True
                st.session_state.pop("experiment2_followup_choice", None)

    st.markdown("#### トラブルシューティング")
    cols1 = st.columns([2, 1])
    with cols1[0]:
        st.markdown("**🤔「実行します」のあとロボットの実行が始まらない場合→**")
    with cols1[1]:
        if st.button("▶️実行を始める", key="manual_request_next_plan"):
            next_plan_request = "行動計画も出力して"
            context.append({"role": "user", "content": next_plan_request})
            st.chat_message("user").write(next_plan_request)
            st.session_state.trigger_llm_call = True
            st.rerun()
    cols2 = st.columns([2, 1])
    with cols2[0]:
        st.markdown("**🚨バグが起きた場合（LLMからの回答がない等）→**")
    with cols2[1]:
        if st.button("⚠️会話をリセット", key="reset_conv"):
            save_conversation_history_to_firestore(
                "会話をリセットしました",
                metadata={"experiment_page": PROMPT_GROUP},
            )
            _reset_conversation_state(system_prompt)
            st.rerun()
    cols = st.columns([2, 1])
    with cols[0]:
        st.markdown("**😊ロボットとのタスクが完了した場合→**")
    with cols[1]:
        if st.button("✅タスク完了！", key="force_end_button"):
            st.session_state.force_end = True
            st.rerun()
    if st.session_state.get("experiment2_followup_prompt"):
        if NEXT_PAGE:
            if st.button("次の実験へ→", key="followup_no", type="primary"):
                st.session_state["experiment2_followup_prompt"] = False
                st.session_state.pop("experiment2_followup_choice", None)
                _reset_conversation_state(system_prompt)
                st.switch_page(NEXT_PAGE)
        else:
            st.info("お疲れさまでした。これで全ての実験が終了です。")
        # if st.button("🙆‍♂️はい → 実験終了", key="followup_yes", type="primary"):
        #     st.session_state["experiment2_followup_prompt"] = False
        #     st.session_state.pop("experiment2_followup_choice", None)
        #     st.success("実験お疲れ様でした！ご協力ありがとうございました。")
        #     st.balloons()

app()
