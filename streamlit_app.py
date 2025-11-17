import streamlit as st

from pages.consent import require_consent

ACTIVE_PAGE_STATE_KEY = "current_active_page"
ACTIVE_PAGE_VALUE = "instructions"

def app():
    # require_consent(allow_withdrawal=True, redirect_to_instructions=False)
    st.session_state[ACTIVE_PAGE_STATE_KEY] = ACTIVE_PAGE_VALUE
    if st.session_state.get("redirect_to_instruction_page"):
        st.session_state["redirect_to_instruction_page"] = False
    # st.title("LLMATCH Criticデモアプリ")
    st.subheader("CHORDへようこそ！")
    st.markdown("""
    慶應義塾大学理工学部情報工学科/LLMATCH研究員の 吉田馨 です。
    本研究にご協力いただきありがとうございます。 
    """)
    st.markdown("""
    このWebアプリは、 **家庭内ロボットを想定したチャットボットデモアプリ** です。
    LLMを搭載したロボットがタスクを遂行する際に、どのようなコミュニケーションスタイルを取るべきかを研究しています。
    """)
    st.warning("説明会に参加されていない方は、以下の説明動画をご覧ください。")
    st.video("https://www.youtube.com/watch?v=Z6bX6YkQX1o")
    st.write("この動画内で共有している操作マニュアルは、以下のリンクからもご覧いただけます。")
    st.write("👉 [Googleスライドを見る](https://docs.google.com/presentation/d/170fsT62Pm_U1_FMcTsrCM27pVbMOy9_ZlhFZP5KOkxw/edit?usp=sharing)")

    st.info(
        """
         **質問やお問い合わせはこちら**  
        Slack の [@Kaoru Yoshida](https://matsuokenllmcommunity.slack.com/team/U071ML4LY5C) までお願いします。
        """,
        icon="📩"
    )
    
    if st.button("実験を始める", use_container_width=True, type="primary"):
        st.session_state["redirect_to_instruction_page"] = False
        st.session_state.pop("experiment1_scroll_reset_done", None)
        st.session_state.pop("experiment_scroll_reset_done", None)
        st.switch_page("pages/01_logical.py")

app()
