

# CHORD: Collaborative Home-Robot Dialogue

<p align="center">
  <a href="#english">English</a> | <a href="#japanese">日本語</a>
</p>

<a name="japanese"></a>

## 📖 概要
**CHORD** (Collaborative Home-Robot Dialogue) は、LLM搭載家庭用ロボットとの協調タスクにおける対話スタイルが、ユーザーとの信頼関係（ラポール）およびタスク遂行に与える影響を検証するために開発されたデモアプリケーションです。

本リポジトリには、実験で使用されたStreamlitアプリケーション、プロンプト設計、および実験関連資料が含まれています。

近年の大規模言語モデル（LLM）の発展により、ロボットは流暢な対話が可能になりました。しかし、家庭内での協調作業において、どのような「対話スタイル」がユーザーとの信頼構築やタスク効率に寄与するかは十分に明らかになっていません。

CHORDは、GPT-4o-miniを搭載したロボットエージェントとチャット形式で対話を行いながら、家事タスク（テーブル準備や花を生けるなど）を共同で行うシミュレーション環境です。

### システム構成
- **Frontend/Backend:** Python / Streamlit
- **LLM:** OpenAI GPT-4o-mini
- **Infrastructure:** Google Cloud Platform (Cloud Run)
- **Database:** Cloud Firestore (状態管理・ログ保存)

## 🔬 研究内容

本システムは、以下の研究の一環として開発・使用されました。

> **論文タイトル:** LLM搭載家庭用ロボットの対話スタイルがラポールとタスク性能に与える影響
> **著者:** 吉田 馨, 山本 匠, 小橋 洋平, 杉浦 裕太

### 実験の目的
LLM搭載ロボットのコミュニケーションスタイル（対話の調子）を変化させた際、ユーザーの**ラポール形成とタスクパフォーマンス**にどのような違いが生まれるかを明らかにすること。

### 比較した3つの対話スタイル
本システムでは、プロンプトエンジニアリングにより以下の3つのスタイルを切り替えて実験を行いました。

1.  **タスク志向型 (Task-oriented)**
    * 効率重視。最小限の情報伝達に徹し、雑談や情緒的反応を避けるスタイル。
2.  **共感型 (Empathetic)**
    * ユーザーへの配慮重視。「素敵ですね」「疲れていませんか？」といった労いや共感の言葉をかけつつ、タスクから逸脱しないスタイル。
3.  **雑談型 (Small-talk)**
    * 親しみやすさ重視。環境内のオブジェクトやユーザーの感情について積極的に話題を広げ、タスクとは無関係な雑談を行うスタイル。

## 🤖 プロンプト設計

本研究では、システムプロンプト内の指示を書き換えることでロボットの人格（対話スタイル）を制御しています。各スタイルの定義は以下の通りです。
※ 実際のプロンプトファイルは `prompts/` ディレクトリを参照してください。

| スタイル | プロンプトの特徴 |
| :--- | :--- |
| **Task-oriented** | ・感情的な言葉を排除する<br>・事実と次のアクションのみを簡潔に伝える<br>・XMLタグやFunction callingの出力形式を厳守する |
| **Empathetic** | ・ユーザーの感情に寄り添う言葉を含める<br>・タスクの進行を妨げない範囲で励ましや賞賛を行う<br>・丁寧で温かみのある口調 |
| **Small-talk** | ・ユーザーの入力に対して関連する話題（天気、インテリア、感情など）を広げる<br>・人間らしい「不完全さ」や「遊び」を持たせる<br>・タスク以外の話題への脱線を許容する |

ロボットは `AvailableSkills` (find, pick up, take 等11項目) からサブタスクを計画し、Function Sequence として出力します。

## 📝 アンケート質問項目

本実験では、以下の指標を用いて評価を行いました。

### NASA-TLX
タスク負荷の評価指標です。

* **精神的要求 (Mental Demand)**
  * あなたは，ロボットと会話をするにあたって，精神的要求（思考，意志決定，計算，記憶，観察，検索，等）がどれくらい要求されましたか？
* **身体的要求 (Physical Demand)**
  * あなたは，ロボットと会話をするにあたって，身体的要求（押す，引く，回す， 操作する等）がどれくらい要求されましたか？
* **時間的切迫感 (Temporal Demand)**
  * あなたは，ロボットと会話をするにあたって，時間的切迫感（作業や要素作業の頻度や速さ）をどの程度感じましたか？
* **作業達成度 (Performance)**
  * ロボットと会話をするにあたって，あなた自身が想定した作業（指示）は，どの程度ロボットによって達成されたと考えますか？
* **努力 (Effort)**
  * あなたはその作業達成率に到達するのに，どのくらい（精神的及び身体的に）努力しましたか？
* **不満 (Frustration)**
  * あなたは，ロボットと会話をするにあたってどのくらい不安，落胆，いらいら，ストレス，不快感を感じましたか？

### Godspeed Questionnaire Series
ロボットに対する印象評価指標です。

#### Anthropomorphism (擬人化)
* Fake 偽物のような -- Natural 自然な
* Machinelike 機械的 -- Humanlike 人間的
* Unconscious 意識を持たない -- Conscious 意識を持っている
* Artificial 人工的 -- Lifelike 生物的
* Moving rigidly ぎこちない動き -- Moving elegantly 洗練された動き

#### Animacy (生物らしさ)
* Dead 死んでいる -- Alive 生きている
* Stagnant 活気のない -- Lively 生き生きとした
* Mechanical 機械的な -- Organic 有機的な
* Inert 不活発な -- Interactive 対話的な
* Apathetic 無関心な -- Responsive 反応のある

#### Likeability (好感度)
* Dislike 嫌い -- Like 好き
* Unfriendly 親しみにくい -- Friendly 親しみやすい
* Unkind 不親切な -- Kind 親切な
* Unpleasant 不愉快な -- Pleasant 愉快な
* Awful ひどい -- Nice 良い

#### Perceived Intelligence (知能の知覚)
* Incompetent 無能な -- Competent 有能な
* Ignorant 無知な -- Knowledgeable 物知りな
* Irresponsible 無責任な -- Responsible 責任のある
* Unintelligent 知的でない -- Intelligent 知的な
* Foolish 愚かな -- Sensible 賢明な

#### Perceived Safety (安全性の知覚)
* Anxious 不安な -- Relaxed 落ち着いた
* Agitated 動揺している -- Calm 冷静な
* Quiescent 平穏な -- Surprised 驚いた

### Trust Scale (信頼尺度)
以下の項目について，同意できる度合いを回答として得ました。

1. このロボットは能力が高いと信じる
2. 私はこのロボットを信頼している
3. このロボットの助言（アドバイス）は信頼できる
4. 私はこのロボットに頼れる
5. このロボットの動作（ふるまい）は一貫していると思う
6. このロボットの助言に従うとき，このロボットは最善を尽くしてくれると信頼している

## 🙌 謝辞

本調査には、**東京大学松尾・岩澤研究室 LLM コミュニティのプログラム LLMATCH** にご協力頂きました。
ここに深く感謝の意を表します。

## 📚 参考文献
* [1] 吉田 馨, 山本 匠, 小橋 洋平, 杉浦 裕太: "LLM搭載家庭用ロボットの対話スタイルがラポールとタスク性能に与える影響", Interaction 2026.
* [2] Merritt, S. M.: Affective processes in human-automation interactions, Human Factors, Vol. 53, No. 4, pp. 356-370 (2011).
* [3] Bartneck, C. et al.: Measurement instruments for the anthropomorphism, animacy, likeability, perceived intelligence, and perceived safety of robots, International Journal of Social Robotics, Vol. 1, No. 1, pp. 71-81 (2009).

---

<a name="english"></a>

## 📖 Overview

**CHORD** (Collaborative Home-Robot Dialogue) is a demo application designed to investigate how the dialogue style of an LLM-equipped home robot affects user rapport and task performance.

This repository contains the Streamlit application, prompt designs, and experimental materials used in the research.

Recent advancements in Large Language Models (LLMs) have enabled robots to engage in fluent conversation. However, it remains unclear what "dialogue style" effectively builds trust and enhances efficiency in collaborative home tasks.

CHORD provides a simulation environment where a user and a robot agent (powered by **GPT-4o-mini**) collaborate via chat to complete domestic chores (e.g., setting a table, arranging flowers).

### Tech Stack
- **Frontend/Backend:** Python / Streamlit
- **LLM:** OpenAI GPT-4o-mini
- **Infrastructure:** Google Cloud Platform (Cloud Run)
- **Database:** Cloud Firestore (State management & Logging)

## 🔬 Research Description

This system was developed for the following research:

> **Paper:** *Effects of Dialogue Styles on Rapport and Task Performance in LLM-equipped Home Robots.*
> **Authors:** Kaoru Yoshida, Takumi Yamamoto, Yohei Kobashi, Yuta Sugiura

### Objective
To reveal how differences in a robot's communication style influence **rapport formation (trust/familiarity)** and **task performance (efficiency/load)** during collaborative tasks.

### Dialogue Styles (Experimental Conditions)
We manipulated the robot's personality using prompt engineering to compare three distinct styles:

1.  **Task-oriented**
    * Prioritizes efficiency. Focuses strictly on minimal information transfer required for the task, avoiding small talk or emotional reactions.
2.  **Empathetic**
    * Prioritizes user consideration. Offers empathetic remarks (e.g., "That looks lovely," "Are you tired?") and encouragement without deviating from the task.
3.  **Small-talk**
    * Prioritizes friendliness. Actively initiates topics about the environment or user's feelings, allowing for "human-like imperfections" and casual conversation unrelated to the task.

## 🤖 Prompt Design

The robot's dialogue style is controlled by system prompts. The table below summarizes the definitions used for each style:
*(See `prompts/` directory for full prompt files.)*

| Style | Key Characteristics |
| :--- | :--- |
| **Task-oriented** | - Eliminate emotional language.<br>- State facts and next actions concisely.<br>- Strictly follow XML tags and function calling formats. |
| **Empathetic** | - Include words that align with the user's emotions.<br>- Provide praise and care without hindering task progress.<br>- Maintain a polite and warm tone. |
| **Small-talk** | - Expand on topics related to user input (e.g., interior design, feelings).<br>- Allow for human-like "playfulness" or "imperfection."<br>- Permit digressions from the main task. |

The robot plans sub-tasks from a defined `AvailableSkills` set (e.g., *find, pick up, take*) and outputs them as a Function Sequence.

## 📝 Evaluation Metrics

In our experiment, we evaluated the system using the following metrics:

### 1. Robot Performance
* **Task Efficiency:** Calculated from task completion time.
* **Subjective Performance:** NASA-TLX (Performance subscale).

### 2. Cognitive Load (NASA-TLX)
Measured using the standard 6 subscales (Mental Demand, Physical Demand, Temporal Demand, Performance, Effort, Frustration).

### 3. Trust Scale
Adopted Merritt's *Trust Scale* for human-automation interaction.
* Competence / Trust
* Reliability of advice
* Consistency of behavior, etc.

### 4. Robot Impression (Godspeed Questionnaire)
Adopted all 5 indices from the *Godspeed Questionnaire*:
* **Anthropomorphism** (Fake ↔ Natural, Machine-like ↔ Human-like)
* **Animacy** (Dead ↔ Alive, Stagnant ↔ Lively)
* **Likeability** (Unpleasant ↔ Pleasant, Unfriendly ↔ Friendly)
* **Perceived Intelligence** (Incompetent ↔ Competent, Ignorant ↔ Knowledgeable)
* **Perceived Safety** (Anxious ↔ Calm)

## 🙌 Acknowledgments

This research was supported by the **LLMATCH** program of the **Matsuo-Iwasawa Lab, The University of Tokyo**.

## 📚 References
* [1] Yoshida, K., Yamamoto, T., Kobashi, Y., Sugiura, Y.: "Impact of Dialogue Styles of LLM-equipped Home Robots on Rapport and Task Performance", Interaction 2026.
* [2] Merritt, S. M.: Affective processes in human-automation interactions, Human Factors, Vol. 53, No. 4, pp. 356-370 (2011).
* [3] Bartneck, C. et al.: Measurement instruments for the anthropomorphism, animacy, likeability, perceived intelligence, and perceived safety of robots, International Journal of Social Robotics, Vol. 1, No. 1, pp. 71-81 (2009).

*This repository implements the experimental system described in the paper above.*
