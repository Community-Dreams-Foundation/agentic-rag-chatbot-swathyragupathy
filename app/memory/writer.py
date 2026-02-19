"""Selective memory: decide whether to write user/company fact and append to markdown files."""
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import OPENAI_API_KEY, OPENAI_MODEL, USER_MEMORY_PATH, COMPANY_MEMORY_PATH


def maybe_write_memory(user_message: str, conversation_summary: Optional[str] = None) -> List[Dict[str, str]]:
    """
    If the user message contains a high-signal fact (preference, role, org learning), return list of
    {target: "USER"|"COMPANY", summary: "..."}. Otherwise return [].
    Caller then calls append_memory for each and appends to USER_MEMORY.md / COMPANY_MEMORY.md.
    """
    if not OPENAI_API_KEY or not user_message.strip():
        return []

    system = """You are a memory classifier for a chatbot. Given a user message, decide if it contains a fact worth storing.
- USER: store only user-specific facts (e.g. "I prefer weekly summaries on Mondays", "I'm a Project Finance Analyst"). One concise sentence.
- COMPANY: store only org-wide, reusable learnings (e.g. "Asset Management interfaces with Project Finance", "Recurring bottleneck is X"). One concise sentence.
- Do NOT store: greetings, questions, vague statements, secrets, PII, or raw conversation.
Output a JSON array of objects with keys: target ("USER" or "COMPANY"), summary (string). If nothing to store, output []."""
    user = f"User message: {user_message}"
    if conversation_summary:
        user += f"\n(Recent context: {conversation_summary})"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown code block if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        import json
        items = json.loads(text)
        if not isinstance(items, list):
            return []
        out = []
        for x in items:
            if isinstance(x, dict) and x.get("target") in ("USER", "COMPANY") and x.get("summary"):
                out.append({"target": x["target"], "summary": str(x["summary"]).strip()})
        return out
    except Exception:
        return []


def append_memory(target: str, summary: str) -> None:
    """Append one line to USER_MEMORY.md or COMPANY_MEMORY.md. target is 'USER' or 'COMPANY'."""
    if target == "USER":
        path = USER_MEMORY_PATH
    elif target == "COMPANY":
        path = COMPANY_MEMORY_PATH
    else:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# USER MEMORY\n\n" if target == "USER" else "# COMPANY MEMORY\n\n", encoding="utf-8")
    content = path.read_text(encoding="utf-8")
    line = f"- {summary}\n"
    if line.strip() not in content:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
