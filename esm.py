import ast
import re

class ExternalStateManager:
    def __init__(self):
        # 1. 既知の初期状態
        self.current_state = {
            "robot_status": {
                "location": "living_room",
                "holding": None
            },
            "environment": {
            "キッチンの棚": [
                "皿",
                "サラダボウル",
                "お椀",
                "木製のボウル",
                "どんぶり",
                "小皿",
                "コップ",
                "ワイングラス",
                "湯呑み",
                "マグカップ",
                "急須",
                "ティーポット",
                "タッパー（保存容器）",
                "コーヒーメーカー",
                "調味料入れ"
            ],
            "キッチンの引き出し": [
                "箸",
                "スプーン",
                "ナイフ",
                "フォーク",
                "おたま",
                "フライ返し",
                "菜箸",
                "ピーラー",
                "栓抜き",
                "缶切り",
                "計量スプーン",
                "ハサミ",
                "剣山",
                "ラップ",
                "アルミホイル",
                "キッチンペーパー"
            ],
            "ダイニングテーブル": [
                "ティッシュ",
                "花瓶の花",
                "ノートパソコン",
                "テーブルクロス",
                "カトラリーケース",
                "リモコン",
                "郵便物",
                "卓上醤油（調味料）"
            ],
            "リビングルーム": [
                "ソファ",
                "クッション",
                "ローテーブル",
                "本棚",
                "本",
                "雑誌",
                "観葉植物",
                "エアコン",
                "空気清浄機",
                "テレビ",
                "DVD",
                "テレビゲーム",
                "ゲーム機本体",
                "スピーカー",
                "Wi-Fiルーター",
                "フロアランプ",
                "ラグ（カーペット）",
                "カーテン",
                "時計",
                "花束"
            ],
            "一番上の棚": [
                "花瓶",
                "アルバム",
                "箱（収納ボックス）",
                "トロフィー",
                "あまり使わない本"
            ],
            "クローゼット": [
                "スーツ",
                "シャツ",
                "Tシャツ",
                "セーター",
                "ズボン",
                "ジーンズ",
                "スカート",
                "ワンピース",
                "コート",
                "パジャマ",
                "キャップ",
                "帽子",
                "ネクタイ",
                "ベルト",
                "バッグ",
                "靴下",
                "下着"
            ],
            "物置": [
                "新聞",
                "スーツケース",
                "ゴルフバッグ",
                "ヒーター",
                "扇風機",
                "掃除機",
                "工具箱",
                "防災グッズ",
                "季節の飾り",
                "キャンプ用品",
                "使わない家電"
            ],
            "玄関": [
                "靴",
                "傘",
                "傘立て",
                "靴箱",
                "スリッパ",
                "鍵",
                "印鑑",
                "宅配ボックス",
                "靴べら"
            ],
            "洗面所": [
                "歯ブラシ",
                "歯磨き粉",
                "コップ",
                "タオル",
                "ハンドソープ",
                "洗顔料",
                "鏡",
                "洗濯機",
                "洗剤",
                "柔軟剤",
                "ドライヤー",
                "体重計"
            ],
            "浴室": [
                "シャンプー",
                "コンディショナー",
                "ボディソープ",
                "風呂椅子",
                "洗面器",
                "バスマット",
                "スポンジ",
                "カミソリ",
                "風呂の蓋",
                "掃除用ブラシ"
            ],
            "トイレ": [
                "トイレットペーパー",
                "トイレブラシ",
                "芳香剤",
                "サニタリーボックス",
                "便座カバー",
                "掃除用シート"
            ],
            "寝室": [
                "ベッド",
                "布団",
                "枕",
                "目覚まし時計",
                "サイドテーブル",
                "間接照明（ランプ）",
                "加湿器",
                "鏡台（ドレッサー）",
                "洋服ダンス"
            ],
            "デスク": [
                "デスクトップパソコン",
                "モニター",
                "キーボード",
                "マウス",
                "卓上ランプ",
                "ペン",
                "ペン立て",
                "ノート",
                "書類",
                "充電ケーブル",
                "ヘッドホン",
                "プリンター",
                "本"
            ],
            "冷蔵庫": [
                "牛乳",
                "卵",
                "野菜",
                "果物",
                "飲み物",
                "お茶",
                "バター",
                "ヨーグルト",
                "ケチャップ",
                "マヨネーズ",
                "冷凍食品",
                "氷"
            ],
            "ベランダ": [
                "洗濯物干し",
                "洗濯バサミ",
                "植木鉢",
                "サンダル",
                "室外機"
            ]
            },
            "task_goal": {
                "target_location": None,
                "items_needed": {}
            }
        }
    
    def set_task_goal_from_llm(self, goal_description_from_llm):
        """
        [フェーズ1: サブゴール設定]
        LLMとの対話で決定した「タスク目標」をパースして、self.current_state に格納する。
        LLMが "Goal: {target: 'dining_table', items: {'plate': 2}}" のようなJSONを出力
        """
        try:
            # "Goal: " の後の辞書部分 {...} を正規表現で抽出
            match = re.search(r'Goal:\s*(\{.*\})', goal_description_from_llm, re.DOTALL)
            if not match:
                print(f"Error: Could not find 'Goal: {{...}}' pattern in: {goal_description_from_llm}")
                return False

            goal_str = match.group(1)
            # 文字列を安全にPythonの辞書に変換
            goal_dict = ast.literal_eval(goal_str) 
            
            self.current_state['task_goal']['target_location'] = goal_dict.get('target_location')
            self.current_state['task_goal']['items_needed'] = goal_dict.get('items_needed', {})
            print(f"Goal Set: {self.current_state['task_goal']}")
            return True # パース成功
        except Exception as e:
            print(f"Error parsing task goal: {e}")
            print(f"Received string: {goal_description_from_llm}")
            return False # パース失敗

    def get_state_as_xml_prompt(self):
        """
        [フェーズ2: 計画]
        現在の self.current_state を、LLMのプロンプトに埋め込むためのXML形式に変換する
        """
        state = self.current_state
        xml_prompt = "<CurrentState>\n"
        xml_prompt += f"  <RobotStatus>\n"
        xml_prompt += f"    <Location>{state['robot_status']['location']}</Location>\n"
        xml_prompt += f"    <Holding>{state['robot_status']['holding']}</Holding>\n"
        xml_prompt += f"  </RobotStatus>\n"
        xml_prompt += f"  <Environment>\n"
        xml_prompt += f"    <dining_table>{state['environment']['dining_table']}</dining_table>\n"
        xml_prompt += f"    <kitchen_shelf>{state['environment']['kitchen_shelf']}</kitchen_shelf>\n"
        xml_prompt += f"    <kitchen_drawer>{state['environment']['kitchen_drawer']}</kitchen_drawer>\n"
        xml_prompt += f"    <living_room>{state['environment']['living_room']}</living_room>\n"
        xml_prompt += f"    <top_shelf>{state['environment']['top_shelf']}</top_shelf>\n"
        xml_prompt += f"    <closet>{state['environment']['closet']}</closet>\n"
        xml_prompt += f"    <storeroom>{state['environment']['storeroom']}</storeroom>\n"
        xml_prompt += f"  </Environment>\n"
        xml_prompt += f"  <TaskGoal>\n"
        xml_prompt += f"    <TargetLocation>{state['task_goal']['target_location']}</TargetLocation>\n"
        xml_prompt += f"    <ItemsNeeded>{state['task_goal']['items_needed']}</ItemsNeeded>\n"
        xml_prompt += f"  </TaskGoal>\n"
        xml_prompt += "</CurrentState>"
        return xml_prompt
    
    def update_state_from_action(self, executed_action_string):
        """
        [フェーズ3: 実行と更新]
        LLMが生成した計画（の1ステップ）を実行した後に呼び出される。
        実行された行動（文字列）をパースし、seld.current_state を更新する。
        これが「状態の逐次更新」の核となる部分です。
        """
        log_messages = []

        def log(message):
            print(message)
            log_messages.append(message)

        log(f"Action Executed: {executed_action_string}")
        action = executed_action_string.lower().strip()
        state = self.current_state

        try:
            if action.startswith("go to the"):
                location = action.replace("go to the", "").strip()
                state["robot_status"]["location"] = location
                log(f"Robot moved to {location}")
            
            elif action.startswith("pick up the"):
                item = action.replace("pick up the", "").strip()
                location = state["robot_status"]["location"]
                if item in state["environment"].get(location, []):
                    state["environment"][location].remove(item)
                    state["robot_status"]["holding"] = item
                    log(f"Robot picked up {item} from {location}")
                else:
                    log(f"Item {item} not found at {location}")

            elif action.startswith("put down the"):
                item = action.replace("put down the", "").strip()
                location = state["robot_status"]["location"]
                if state["robot_status"]["holding"] == item:
                    state["environment"].setdefault(location, []).append(item)
                    state["robot_status"]["holding"] = None
                    log(f"Robot put down {item} at {location}")
                else:
                    log(f"Robot is not holding {item}")

            elif action.startswith("find a/an"):
                item = action.replace("find a", "").replace("find an", "").strip()
                found = False
                for loc, items in state["environment"].items():
                    if item in items:
                        log(f"Found {item} at {loc}")
                        found = True
                        break
                if not found:
                    log(f"{item} not found in the environment")

            elif action.startswith("open the drawer"):
                log("Robot opened the drawer (no state change implemented)")
            
            elif action.startswith("close the drawer"):
                log("Robot closed the drawer (no state change implemented)")
            
            elif "in the drawer" in action and action.startswith("put"):
                item_match = re.search(r'put (.*?) in the drawer', action)
                if item_match:
                    item = item_match.group(1).strip()
                    location = state["robot_status"]["location"]
                    if state["robot_status"]["holding"] == item:
                        state["environment"].setdefault(location, []).append(item)
                        state["robot_status"]["holding"] = None
                        log(f"Robot put {item} in the drawer at {location}")
                    else:
                        log(f"Robot is not holding {item}")

            elif "out of the drawer" in action and action.startswith("take"):
                item_match = re.search(r'take (.*?) out of the drawer', action)
                if item_match:
                    item = item_match.group(1).strip()
                    location = state["robot_status"]["location"]
                    if item in state["environment"].get(location, []):
                        state["environment"][location].remove(item)
                        state["robot_status"]["holding"] = item
                        log(f"Robot took {item} out of the drawer at {location}")
                    else:
                        log(f"Item {item} not found in the drawer at {location}")
            
            elif action.startswith("done"):
                log("Task completed.")

            else:
                log(f"Unrecognized action: {action}")

        except Exception as e:
            log(f"State Update Error: {e} on action: {action}")
            # (失敗した場合の処理)

        log(
            f"State Updated: Robot at {state['robot_status']['location']}, holding {state['robot_status']['holding']}"
        )

        return "\n".join(log_messages)
