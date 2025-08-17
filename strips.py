import streamlit as st
import re

TAG_RE = re.compile(r"</?([A-Za-z0-9_]+)(\s[^>]*)?>")

def strip_tags(text: str) -> str:
    return TAG_RE.sub("", text or "").strip()

def extract_between(tag: str, text: str) -> str | None:
    m = re.search(fr"<{tag}>([\s\S]*?)</{tag}>", text or "", re.IGNORECASE)
    return m.group(1).strip() if m else None

def parse_step(step):
    # <move_to>BEDROOM</move_to> → move_to("BEDROOM")
    m = re.match(r"<(\w+)>(.*?)</\1>", step.strip())
    if m:
        func = m.group(1)
        arg = m.group(2).strip()
        # 引数がカンマ区切りの場合も考慮
        args = [a.strip() for a in arg.split(",")] if "," in arg else [arg]
        args_str = ", ".join([f'"{a}"' for a in args])
        return f"{func}({args_str})"
    return step  # それ以外はそのまま
