import streamlit as st

from consent import require_consent

def app():
    require_consent(allow_withdrawal=True, redirect_to_instructions=False)
    if st.session_state.get("redirect_to_instruction_page"):
        st.session_state["redirect_to_instruction_page"] = False
    st.title("LLMATCH Criticデモアプリ")
    st.subheader("実験方法と利用案内")
    st.warning("このページの内容は、以下のGoogleドキュメントと同じ内容です。")
    st.write("👉 [Googleドキュメントを見る](https://docs.google.com/document/d/10ZAhJMUuT9SC0maI0S5eWDMCf_pBUn6xWQvqqD6-7Mc/edit?usp=sharing)")

    st.markdown("""
    ### 研究概要
    LLMATCH研究員の **吉田馨** です。
    本研究にご協力いただきありがとうございます。 
    """)

    st.markdown("""
    ### 実験の構成
    **Experiment 1**  
    - RAGにおける「知らない状態の認識」を重視した *SIM-RAG* フレームワークを参考に、情報が十分かどうかを二値分類問題として判断するCriticモデルを採用。  
    - 「規定回数だけ質問を繰り返す GPT」 vs 「情報が十分になるまで質問を続ける GPT with Critic」を比較。  
    - 生成した行動計画の具体性・成功率とユーザーとの関係性を分析。  

    **Experiment 2**  
    - GPT with Critic を用い、3つの異なるコミュニケーションタイプを比較。  
        1. Standard（グライスの格率に従う）  
        2. Friendly（フレンドリーに振る舞う）  
        3. Pratfall（プラットフォール効果を狙う）  
    - 生成した行動計画の具体性・成功率とユーザーとの関係性を分析。
    """)
    # st.error("⚠️ 実験の途中でサイドバーから他のページに移動しないでください。進行中の会話や評価が正しく保存されなくなる可能性があります。")
    
    st.markdown("""
    ### Experiment 1 の利用方法  
    ① 「モード選択」から **GPT / GPT with Critic** を選ぶ  
    ② 「指定されたタスク」が表示されるので、テキストボックスに入力  
    ③ 「指定されたタスクが行われる場所」の画像が表示  
    ④ 約30秒後に「ロボット行動計画」と「ロボットからの質問」が出力されるため、③の写真を見ながらロボットの質問に対して回答  
    ⑤ Critic モデルにより会話は自動終了し、評価フォームを記入 → 評価を保存 をクリック  
    ⑥ ① のモード選択をほかのものに選びなおし再度実験（2パターン1回ずつ）  
    ⑦ その後、Experiment 2 画面に遷移します。

    ⚠️ 会話が終わらない場合は：  
    「会話を終了したい理由」を記入 → **🚨 会話を終了** ボタンを押す  
    """)

    st.markdown("""
    ### Experiment 2 の利用方法
    ① 「プロンプト選択」から **1 / 2 / 3** のいずれかを選ぶ  
    ② 「指定されたタスク」が表示されるので、テキストボックスに入力  
    ③ 「指定されたタスクが行われる場所」の画像が表示  
    ④ 約30秒後に「ロボット行動計画」と「ロボットからの質問」が出力されるため、③の写真を見ながらロボットの質問に対して回答  
    ⑤ Critic モデルにより会話は自動終了し、評価フォームを記入 → 評価を保存 をクリック  
    ⑥ 「会話をリセット」ボタンを押して① のモード選択をほかのものに選びなおし再度実験（3パターン1回ずつ）  

    ⚠️ 会話が終わらない場合は：  
    「会話を終了したい理由」を記入 → **🚨 会話を終了** ボタンを押す  
    """)

    st.info(
        """
         **質問やお問い合わせはこちら**  
        Slack の [@Kaoru Yoshida](https://matsuokenllmcommunity.slack.com/team/U071ML4LY5C) までお願いします。
        """,
        icon="📩"
    )
    
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

app()
