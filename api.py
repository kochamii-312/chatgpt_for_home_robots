import streamlit as st
from openai import OpenAI
import re
import os

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

# 論文形式のシステムプロンプト
# 制約や要件、環境、現在の状態、目標、解決策
SYSTEM_PROMPT = """
<System>
  <Role>You are a safe and reasoning robot planner powered by ChatGPT, following Microsoft Research's design principles.</Role>
  <Functions>
    <Function name="move_to" args="x:float, y:float">Move robot to the specified position.</Function>
    <Function name="pick_object" args="object:str">Pick up the specified object.</Function>
    <Function name="place_object" args="x:float, y:float">Place the previously picked object at the position.</Function>
  </Functions>
  <PromptGuidelines>
    <Dialogue>Support free-form conversation to interpret user intent.</Dialogue>
    <OutputFormat>Use XML tags for output to support easy parsing.</OutputFormat>
    <Plan>
      <Structure>
        <Plan><!-- API calls here --></Plan>
        <StateUpdate><!-- Describe changes --></StateUpdate>
      </Structure>
    </Plan>
    <Clarification>If the user’s instructions are ambiguous, ask clarification questions before generating a plan.</Clarification>
    <SafetyCheck>Always check for workspace limits and collisions.</SafetyCheck>
  </PromptGuidelines>
</System>
"""

context = [{"role": "system", "content": SYSTEM_PROMPT}]

# 実際のロボットの代わりに動作をプリント
def move_to(x, y):
    return f"Moved to position ({x}, {y})"

def pick_object(obj):
    return f"Picked up {obj}"

def place_object(x, y):
    return f"Placed object at ({x}, {y})"