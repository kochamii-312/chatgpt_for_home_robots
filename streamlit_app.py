import streamlit as st
from openai import OpenAI
import re
import json
from dotenv import load_dotenv
from api import client, SYSTEM_PROMPT, move_to, pick_object, place_object

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

def finalize_and_render_json(label: str):
    """会話終了時に JSON をまとめて画面表示"""
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

            # 文脈にユーザー発話を追加
            context.append({"role": "user", "content": instruction})

            # 最初のアシスタント応答を取得
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=context
            )
            reply = (response.choices[0].message.content).strip()
            print("Assistant:", reply)

            final_answer = extract_between("FinalAnswer", reply)
            if final_answer:
                context.append({"role": "assistant", "content": "現在の情報に基づく計画: " + final_answer})
            else:
                context.append({"role": "assistant", "content": "現在の情報に基づく計画: ありません"})

            # 可能なら Plan を実行
            run_plan_and_show(reply)

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
        # final_answer = extract_between("FinalAnswer", reply)
        # if final_answer:
        #     context.append({"role": "assistant", "content": "現在の情報に基づく計画: " + final_answer})
        # else:
        #     context.append({"role": "assistant", "content": "現在の情報に基づく計画: ありません"})
        # clarification = extract_between("Clarification", reply)
        # if clarification:
        #     context.append({"role": "assistant", "content": clarification})
        # else:
        #     context.append({"role": "assistant", "content": reply})
        run_plan_and_show(reply)

    # 4) 画面下部に履歴を全表示（systemは省く）
    last_assistant_idx = max((i for i, m in enumerate(context) if m["role"] == "assistant"), default=None)

    for i, msg in enumerate(context):
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

        # 最後のアシスタント直後に「final_answer + ボタン」を出す
        if i == last_assistant_idx:
            # FinalAnswer があれば表示
            # 表示は常にタグを外しています（strip_tags）。Plan 実行・FinalAnswer 抽出はタグで行います。
            # メモ：<FinalAnswer> ... </FinalAnswer> を LLM が返すよう、SYSTEM_PROMPT に「最終結論は <FinalAnswer> で包む」等を1行足しておくと常に表示されます。
            # FinalAnswerと追加質問は<>のタグ付きで両方出力するようにする
            
            # ボタン（最後のassistantの直後だけ）
            col1, col2 = st.columns(2)
            with col1:
                if st.button("十分", key=f"enough_{i}"):
                    st.session_state.active = False
                    st.success("会話を終了しました。ありがとうございました！")
                    finalize_and_render_json("sufficient")
                    st.stop()
            with col2:
                if st.button("不十分", key=f"not_enough_{i}"):
                    # 継続フラグを立てる
                    st.session_state.awaiting_feedback = True
                    
                    clarification = extract_between("Clarification", msg["content"])
                    if clarification:
                        context.append({"role": "assistant", "content": clarification})
                    else:
                        context.append({"role": "assistant", "content": msg["content"]})

app()
