"""Streamlit helpers for obtaining participant consent before running the app."""

from __future__ import annotations

import streamlit as st


CONSENT_TEXT = """
### 研究への参加と個人情報の取り扱いについて

本研究は、人間とAIとの対話が家庭用ロボットの行動計画に与える影響を明らかにすることを目的としています。本アプリでは、指示文とロボットの行動計画、ユーザーとの対話内容に関する記録を取得します。

#### 1. 参加の任意性と撤回の自由
- 参加は完全に任意であり、同意後であっても不利益なく撤回できます。
- 撤回したい場合は、サイドバー（左上の≫クリックで表示）のボタンでいつでも中止してください。

#### 2. 研究の実施内容
- 画面に表示されるタスクに関して、家庭用ロボットに与える指示を入力すると、AIから質問が来ます。これに回答することにより、AIとの対話を通じて、AIが自動的に行動計画を作成します。
- 所要時間は全部で約15〜20分を想定していますが、途中で中止しても問題ありません。

#### 3. 取得するデータ
- 入力されたテキスト、会話ログ、アプリ操作に関する記録を保存します。
- 取得データから個人が特定される恐れがある情報を入力しないようご留意ください。

#### 4. 個人情報とプライバシーの保護
- 収集データは匿名化した上で安全に保管し、研究目的以外には利用しません。
- 研究成果を学会等で発表する際も、個人が特定される形で情報を公開することはありません。

#### 5. 期待される利益と潜在的な不利益
- 参加者には、直接的な金銭的報酬や利益はありません。家庭用ロボットのユーザー体験向上に資する研究に貢献できます。
- 実験中、AIとの対話内容によってストレスや不快感を覚える可能性があります。不快に感じた場合は速やかに参加を中止してください。

#### 6. 利益相反に関する情報
- 本研究は LLMATCH 内で実施され、研究員は関連企業からの資金提供や利害関係を有していません。
- 研究成果は学術的知見として公開されますが、特定企業の宣伝を目的としていません。

#### 7. お問い合わせ
- 研究内容やデータの取り扱いに関するご質問は、Slack の [@Kaoru Yoshida](https://matsuokenllmcommunity.slack.com/team/U071ML4LY5C) までお問い合わせください。

上記の説明を読み、内容を理解した上で参加に同意する場合は、以下のチェックボックスをオンにしてボタンを押してください。
"""


def _render_consent_form() -> None:
    """Render the consent form and stop execution until the user agrees."""

    st.set_page_config(page_title="研究参加に関する同意", layout="wide")
    st.title("研究参加の同意について")
    st.markdown(CONSENT_TEXT)

    with st.form("consent_form", clear_on_submit=False):
        agree = st.checkbox("上記の説明を読み、研究に参加することに同意します。", value=False)
        submit = st.form_submit_button("同意して実験に進む", use_container_width=True)

    if submit:
        if agree:
            st.session_state["consent_given"] = True
            st.session_state["redirect_to_instruction_page"] = True
            st.success("ご同意ありがとうございます。実験画面に進みます。")
            st.rerun()
        else:
            st.error("参加には同意が必要です。チェックボックスにチェックを入れてください。")

    st.stop()


def require_consent(
    *, allow_withdrawal: bool = False, redirect_to_instructions: bool = True
) -> None:
    """Ensure that the participant has given consent before proceeding."""

    if not st.session_state.get("consent_given"):
        _render_consent_form()

    if redirect_to_instructions and st.session_state.get("redirect_to_instruction_page"):
        st.session_state["redirect_to_instruction_page"] = False
        st.switch_page("streamlit_app.py")

    if allow_withdrawal:
        with st.sidebar:
            if st.button("同意を撤回してトップに戻る", type="secondary"):
                st.session_state["consent_given"] = False
                st.rerun()
