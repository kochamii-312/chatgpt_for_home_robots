import streamlit as st

from consent import require_consent

def app():
    require_consent(allow_withdrawal=True)
    st.title("LLMATCH Criticデモアプリ")
    st.subheader("実験方法と利用案内")
    st.warning("このページの内容は、以下のGoogleドキュメントと同じ内容です。")
    st.write("👉 [Googleドキュメントを見る](https://docs.google.com/document/d/10ZAhJMUuT9SC0maI0S5eWDMCf_pBUn6xWQvqqD6-7Mc/edit?usp=sharing)")

    st.markdown("""
    ### 研究概要
    LLMATCH研究員の **吉田馨** です。私は「人間とAIの関係性がロボットの最終的なタスク達成率に影響を与える」という仮説を検証するために、このデモアプリを用いた実験を行います。  
    この研究には実際に利用した方々のデータが数多く必要です。ご協力をお願いいたします。

    本デモアプリは「家庭用ロボットにタスクを指示し、対話を通じて行動計画を生成する」仮想システムを模したものです。
    ユーザーがLLMに指示を入力すると、LLMが指示を実行するために不足している情報について質問し、家庭用ロボットの行動計画を更新していきます。
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
    st.error("⚠️ 実験の途中でサイドバーから他のページに移動しないでください。進行中の会話や評価が正しく保存されなくなる可能性があります。")
    
    st.markdown("""
    ### Experiment 1 の利用方法
    1. サイドバーから **Experiment 1** を開く  
    2. 「モード選択」から **GPT / GPT with Critic** を選ぶ  
    3. 「評価モデル」は選択不要  
    4. 家の画像をもとに「家 → 部屋 → 表示する画像」を選ぶ（複数選択可）  
    5. 「タスク」が表示されるので、テキストボックスに入力して会話を開始  
    6. 約30秒後に「ロボットの行動計画」と「質問」が出力される  
    7. 質問に回答しながら会話を進める  
    8. Critic により会話は自動終了  

    ⚠️ 会話が終わらない場合は：  
    「会話を終了したい理由」を記入 → **🚨 会話を強制的に終了** ボタンを押す  

    最後に行動計画とコミュニケーションを評価する評価フォームを記入 → 評価を保存 をクリック
    ※行動計画に使われている関数は左のサイドバーに記載

    """)

    st.markdown("""
    ### Experiment 2 の利用方法
    1. サイドバーから **Experiment 2** を開く  
    2. 「プロンプト」は自動で選択済み  
    3. 「評価モデル」は選択不要  
    4. 家の画像をもとに「家 → 部屋 → 表示する画像」を選ぶ（複数選択可）  
    5. 「タスク」が表示されるので、テキストボックスに入力して会話を開始  
    6. 約30秒後に「ロボットの行動計画」と「質問」が出力される  
    7. 質問に回答しながら会話を進める  
    8. Critic により会話は自動終了  

    ⚠️ 会話が終わらない場合は：  
    「会話を終了したい理由」を記入 → **🚨 会話を強制的に終了** ボタンを押す  

    最後に行動計画とコミュニケーションを評価する評価フォームを記入 → 評価を保存 をクリック
    ※行動計画に使われている関数は左のサイドバーに記載
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
