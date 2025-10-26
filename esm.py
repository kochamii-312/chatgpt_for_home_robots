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
                "kitchen_shelf": ["plate", "salad_bowl", "Japanese_soup_bowl", "wooden_bowl", "glass", "wine_glass", "teacup", "teapot"],
                "kitchen_drawer": ["chopsticks", "spoon", "knife", "fork", "scissors", "flower_frog"],
                "dining_table": ["tissue", "flower in a vase", "laptop"],
                "living_room": ["sofa", "low table", "book shelf", "houseplant", "air conditioner", "air cleaner","TV", "DVDs", "video games", "floor lamp"],
                "top_shelf": ["vase"],
                "closet": ["suits", "shirt", "t-shirt", "skirt", "dress", "coats", "cap", "hat", "bags"],
                "storeroom": ["newspaper", "suitcase", "golf bags", "heater", "electric fan"],
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
        print(f"Action Executed: {executed_action_string}")
        action = executed_action_string.lower().strip()
        state = self.current_state

        try:
            if action.startswith("go to the"):
                location = action.replace("go to the", "").strip()
                state["robot_status"]["location"] = location
                print(f"Robot moved to {location}")
            
            elif action.startswith("pick up the"):
                item = action.replace("pick up the", "").strip()
                location = state["robot_status"]["location"]
                if item in state["environment"].get(location, []):
                    state["environment"][location].remove(item)
                    state["robot_status"]["holding"] = item
                    print(f"Robot picked up {item} from {location}")
                else:
                    print(f"Item {item} not found at {location}")

            elif action.startswith("put down the"):
                item = action.replace("put down the", "").strip()
                location = state["robot_status"]["location"]
                if state["robot_status"]["holding"] == item:
                    state["environment"].setdefault(location, []).append(item)
                    state["robot_status"]["holding"] = None
                    print(f"Robot put down {item} at {location}")
                else:
                    print(f"Robot is not holding {item}")

            elif action.startswith("find a/an"):
                item = action.replace("find a", "").replace("find an", "").strip()
                found = False
                for loc, items in state["environment"].items():
                    if item in items:
                        print(f"Found {item} at {loc}")
                        found = True
                        break
                if not found:
                    print(f"{item} not found in the environment")

            elif action.startswith("open the drawer"):
                print("Robot opened the drawer (no state change implemented)")
            
            elif action.startswith("close the drawer"):
                print("Robot closed the drawer (no state change implemented)")
            
            elif "in the drawer" in action and action.startswith("put"):
                item_match = re.search(r'put (.*?) in the drawer', action)
                if item_match:
                    item = item_match.group(1).strip()
                    location = state["robot_status"]["location"]
                    if state["robot_status"]["holding"] == item:
                        state["environment"].setdefault(location, []).append(item)
                        state["robot_status"]["holding"] = None
                        print(f"Robot put {item} in the drawer at {location}")
                    else:
                        print(f"Robot is not holding {item}")

            elif "out of the drawer" in action and action.startswith("take"):
                item_match = re.search(r'take (.*?) out of the drawer', action)
                if item_match:
                    item = item_match.group(1).strip()
                    location = state["robot_status"]["location"]
                    if item in state["environment"].get(location, []):
                        state["environment"][location].remove(item)
                        state["robot_status"]["holding"] = item
                        print(f"Robot took {item} out of the drawer at {location}")
                    else:
                        print(f"Item {item} not found in the drawer at {location}")
            
            elif action.startswith("done"):
                print("Task completed.")
            
            else:
                print(f"Unrecognized action: {action}")
        
        except Exception as e:
            return(f"State Update Error: {e} on action: {action}")
            # (失敗した場合の処理)
        
        print(f"State Updated: Robot at {state['robot_status']['location']}, holding {state['robot_status']['holding']}")
