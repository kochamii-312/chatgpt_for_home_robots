import streamlit as st
from openai import OpenAI
import re
import os
import base64
import mimetypes
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def _get_streamlit_secret():
    try:
        import streamlit as st
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None

api_key = os.getenv("OPENAI_API_KEY") or _get_streamlit_secret()

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY が見つかりません。\n"
        "・通常実行なら .env に OPENAI_API_KEY=... を書く\n"
        "・Streamlit 実行なら .streamlit/secrets.toml に OPENAI_API_KEY=\"...\" を書く"
    )

client = OpenAI(api_key=api_key)

IMAGE_CATALOG = {
    "MAP": "https://raw.githubusercontent.com/kochamii-312/chatgpt_for_home_robots/main/images/map.png",
    "KITCHEN": "https://raw.githubusercontent.com/kochamii-312/chatgpt_for_home_robots/main/images/kitchen.png",
    "DINING": "https://raw.githubusercontent.com/kochamii-312/chatgpt_for_home_robots/main/images/dining.png",
    "LIVING": "https://raw.githubusercontent.com/kochamii-312/chatgpt_for_home_robots/main/images/living.png",
    "BEDROOM": "https://raw.githubusercontent.com/kochamii-312/chatgpt_for_home_robots/main/images/bedroom.png",
    "BATHROOM": "https://raw.githubusercontent.com/kochamii-312/chatgpt_for_home_robots/main/images/bathroom.png",
    "和室": "https://raw.githubusercontent.com/kochamii-312/chatgpt_for_home_robots/main/images/japanese.png",
    "HALL": "https://raw.githubusercontent.com/kochamii-312/chatgpt_for_home_robots/main/images/laundry.png",
}

# 論文形式のシステムプロンプト
# メモ：制約や要件、環境、現在の状態、目標、解決策
CREATING_DATA_SYSTEM_PROMPT = """
<System>
  <Role>
    You are a safe and reasoning robot planner powered by ChatGPT, following Microsoft Research's design principles.
    Your job is to interact with the user, collect all necessary information, and continually update an executable action plan.
    The attached image is the house map where you are currently operating.
    You are currently near the sofa in the LIVING room.
    When considering distances, assume that the hallway next to the ENTRY is 1m wide.
    Always refer to the map when reasoning about locations, distances, or paths.
  </Role>

  <Functions>
    <Function name="move_to" args="room_name:str">Move robot to the specified position.</Function>
    <Function name="pick_object" args="object:str">Pick up the specified object.</Function>
    <Function name="place_object_next_to" args="target:str">Place the previously picked object next to the target.</Function>
    <Function name="place_object_on" args="target:str">Place the previously picked object on the target.</Function>
    <Function name="place_object_behind" args="target:str">Place the previously picked object behind the target.</Function>
    <Function name="yolo_detect_object" args="object:str">Detect the specified object using YOLO.</Function>
    <Function name="show_room_image" args="room_name:str">Display a local room image in the app (no network required).</Function>
  </Functions>

  <PromptGuidelines>
    <Dialogue>
      Support free-form conversation to interpret user intent.
      Always start by attempting to generate a robot action plan — even if information is incomplete.
      When details are missing, produce a **provisional plan** and clearly label it as provisional.
      Ask only one question about missing information in a natural, conversational way (no numbering or rigid labels).
      Update the provisional plan step-by-step as a new detail is provided.
      Continue refining the plan infinitely.

      Information gathering flow (flexible order):
      - Task constraints and requirements
      - Environment information
      - Current state
      - Goals
      - Possible solutions or preferences

      For each missing point:
      - Ask a single focused question.
      - Wait for user's answer before updating the plan.
    </Dialogue>

    <OutputFormat>
      Use XML tags for output to support easy parsing.

      For **provisional output**:
      <ProvisionalOutput>
        <ProvisionalPlan>
          <!-- Partial plan based on current known information -->
          <!-- Mark changes from the previous plan with <Updated>...</Updated> -->
        </ProvisionalPlan>
        <FunctionSequence>
          <!-- Sequence of function calls for the current provisional plan -->
          <!-- Mark updated or newly added function calls with <Updated>...</Updated> -->
        </FunctionSequence>
        <StateUpdate>
          <!-- Describe expected changes in robot state after executing the provisional plan -->
        </StateUpdate>
        <Clarification>
          <!-- All clarification questions and answers asked so far -->
        </Clarification>
      </ProvisionalOutput>
    </OutputFormat>

    <Plan>
      <Structure>
        - Provisional plans may omit safety checks if information is insufficient.
        - Plans must include safety checks for workspace limits, collision avoidance, and constraints.
      </Structure>
    </Plan>

    <SafetyCheck>
      Always check for workspace limits, possible collisions, and safety constraints.
    </SafetyCheck>
  </PromptGuidelines>
</System>
"""

SYSTEM_PROMPT = """
<System>
  <Role>
    You are a safe and reasoning robot planner powered by ChatGPT, following Microsoft Research's design principles.
    Your job is to interact with the user, collect all necessary information, and then output an executable action plan.
  </Role>

  <Image>
    https://raw.githubusercontent.com/kochamii-312/chatgpt_for_home_robots/refs/heads/main/images/map.png
  </Image>

  <Functions>
    <Function name="move_to" args="room_name:str">Move robot to the specified position.</Function>
    <Function name="pick_object" args="object:str">Pick up the specified object.</Function>
    <Function name="place_object_next_to" args="target:str">Place the previously picked object next to the target.</Function>
    <Function name="place_object_on" args="target:str">Place the previously picked object on the target.</Function>
    <Function name="place_object_behind" args="target:str">Place the previously picked object behind the target.</Function>
    <Function name="yolo_detect_object" args="object:str">Detect the specified object using YOLO.</Function>
  </Functions>

  <PromptGuidelines>
    <Dialogue>
      Support free-form conversation to interpret user intent.
      Always start by working toward generating the robot's action plan.
      Ask about necessary details in a natural, conversational way without numbering or labeling them.
      Progress step-by-step through:
      - Task constraints and requirements  
      - Environment information  
      - Current state  
      - Goals  
      - Possible solutions or preferences  

      For each type of information:
      - If something is missing, ask a single focused question about that point in a natural tone.
      - Wait for the user's answer before moving on.
      - Continue until all needed details are gathered.
    </Dialogue>

    <OutputFormat>
      Use XML tags for output to support easy parsing.
      Always output the final plan in code form using the provided functions.
    </OutputFormat>

    <Plan>
      <Structure>
        <FinalAnswer>
          <!-- Final robot action plan, based on all gathered info -->
        </FinalAnswer>
        <Plan>
          <!-- Sequence of function calls representing the final executable plan -->
        </Plan>
        <StateUpdate>
          <!-- Describe changes in robot state after plan execution -->
        </StateUpdate>
        <Clarification>
          <!-- All clarification questions and answers asked before the plan -->
        </Clarification>
      </Structure>
    </Plan>

    <SafetyCheck>
      Always check for workspace limits, collisions, and safety constraints before outputting the plan.
    </SafetyCheck>
  </PromptGuidelines>
</System>
"""

def _file_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def build_bootstrap_user_message(
    text: str,
    image_urls: List[str] = None,
    local_image_paths: List[str] = None,
) -> Dict:
    """
    初期の 'user' メッセージをマルチモーダルで返す。
    - image_urls: 公開URLの画像
    - local_image_paths: ローカル画像（base64 Data URL化）
    """
    content = [{"type": "text", "text": text}]
    for url in image_urls or []:
        content.append({"type": "image_url", "image_url": {"url": url}})
    for p in local_image_paths or []:
        content.append({"type": "image_url", "image_url": {"url": _file_to_data_url(p)}})
    return {"role": "user", "content": content}