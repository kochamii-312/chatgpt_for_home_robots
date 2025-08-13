from openai import OpenAI
import re

client = OpenAI(api_key="")

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