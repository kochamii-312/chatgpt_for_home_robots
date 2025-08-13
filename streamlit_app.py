import streamlit as st
from openai import OpenAI
import re
import json
import os
from dotenv import load_dotenv
from api import client, context, move_to, pick_object, place_object

load_dotenv()

def app():
    st.title("LLMATCHデモアプリ")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    with st.form(key="instruction_form"):
        st.write("ロボットへの指示")
        instruction = st.text_input("ロボットへの指示")
        submit_btn = st.form_submit_button("実行")
    if submit_btn:
        st.write("ロボットへの指示がセットされました")
        st.session_state["instruction"] = instruction
        context.append({"role": "user", "content": instruction})
        st.session_state["history"].append({"role": "user", "content": instruction})

    # 全履歴を表示
    for msg in st.session_state["history"]:
        message = st.chat_message(msg["role"])
        message.write(msg["content"])

    user_input = st.chat_input("入力してください")
    if user_input:
        context.append({"role": "user", "content": user_input})
        st.session_state["history"].append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context
        )

        reply = response.choices[0].message.content.strip()
        st.session_state["history"].append({"role": "assistant", "content": reply})

        # 〈Plan〉タグを見つけて実行
        plan_match = re.search(r"<Plan>(.*?)</Plan>", reply, re.S)
        if plan_match:
            steps = re.findall(r"<Step>(.*?)</Step>", plan_match.group(1))
            for step in steps:
                try:
                    result = eval(step)
                    print("Result:", result)
                except Exception as e:
                    print("Execution error:", e)


app()
