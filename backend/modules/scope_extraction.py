# modules/scope.py
from __future__ import annotations
import os, json, re
from typing import Dict, Any, List

# Reuse your existing pdf text extraction approach
def _read_file_text(path: str) -> str:
    name = (path or "").lower()
    try:
        if name.endswith(".pdf"):
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            return "\n".join(page.get_text("text") for page in doc)
        elif name.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            # Best effort
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        # Last resort: return empty so we fail fast upstream
        return ""

def _fallback_from_freeform(text: str) -> Dict[str, Any]:
    """
    If the model doesn't give strict JSON, try to salvage something sensible.
    """
    if not text:
        return {"scope_of_work_text": "", "deliverables": []}

    # Try to split on a 'Deliverables' header
    m = re.split(r"\bDeliverables\b[:\n]*", text, maxsplit=1, flags=re.IGNORECASE)
    scope_part = m[0].strip() if m else text.strip()
    deliv_part = m[1].strip() if len(m) > 1 else ""

    # Extract bullets from deliverables part
    deliverables: List[str] = []
    for line in deliv_part.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^([-*•·]|\d+[.)])\s+", line):
            deliverables.append(re.sub(r"^([-*•·]|\d+[.)])\s+", "", line).strip())
        else:
            # treat as bullet anyway if in deliverables block
            deliverables.append(line)
    return {
        "scope_of_work_text": scope_part,
        "deliverables": deliverables,
    }

def extract_scope_and_deliverables_from_file(path: str) -> Dict[str, Any]:
    """
    Reads the document at `path`, asks the LLM to extract:
      - scope_of_work_text: string (may include bullets/newlines)
      - deliverables: list[str]

    RETURNS dict: { "scope_of_work_text": str, "deliverables": [str, ...] }
    """
    from openai import OpenAI

    text = _read_file_text(path)
    if not text.strip():
        # no extractable text, return empty payload
        return {"scope_of_work_text": "", "deliverables": []}

    # keep prompt within a safe token budget
    # trim to ~15k chars (adjust to your model limits)
    trimmed = text[:15000]

    system_msg = (
        "You extract structure from RFQ documents. "
        "Return STRICT JSON with exactly these keys:\n"
        '{"scope_of_work_text": string, "deliverables": string[]} '
        "No commentary, no markdown, no code fences."
    )
    user_msg = (
        "From the RFQ content below, extract:\n"
        "1) scope_of_work_text: a concise narrative (can include bullets/newlines)\n"
        "2) deliverables: a bullet-ready list of items\n\n"
        f"RFQ CONTENT:\n{trimmed}"
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("SCOPE_MODEL", "gpt-4o-mini")

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        out_text = resp.choices[0].message.content or ""

        # Some models wrap JSON in fences; strip them
        out_text = out_text.strip()
        fenced = re.search(r"\{.*\}", out_text, flags=re.DOTALL)
        if fenced:
            out_text = fenced.group(0)

        data = json.loads(out_text)
        # Basic shape enforcement
        scope_text = data.get("scope_of_work_text") or ""
        deliverables = data.get("deliverables") or []
        # Normalize deliverables to list[str]
        deliverables = [str(x).strip() for x in deliverables if str(x).strip()]

        return {
            "scope_of_work_text": scope_text,
            "deliverables": deliverables,
        }
    except Exception:
        # Fall back to heuristic split if JSON parsing or API fails
        return _fallback_from_freeform(locals().get("out_text", ""))
