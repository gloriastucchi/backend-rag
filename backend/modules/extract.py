from __future__ import annotations
import os, json, re
from typing import List, Dict, Any, Tuple
from io import BytesIO

from pypdf import PdfReader
from docx import Document as DocxDocument
import tiktoken
from openai import OpenAI

# add at top
import logging, fitz  # PyMuPDF
logging.basicConfig(level=logging.INFO)

# If you already have a settings module, feel free to import from there
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "gpt-4.1-mini")  # or gpt-4o-mini
MAX_TOKENS_PER_CHUNK = int(os.getenv("EXTRACT_MAX_TOKENS", "2800"))  # safety budget for chat
CHUNK_OVERLAP = int(os.getenv("EXTRACT_CHUNK_OVERLAP", "200"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")

client = OpenAI(api_key=OPENAI_API_KEY)
enc = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT = """You are an expert RFP analyst.
Extract ONLY specific, actionable requirements that are explicitly stated in the provided text.

CRITICAL RULES:
- ONLY extract requirements that are explicitly written (no assumptions).
- DO NOT add generic requirements or common practices.
- Each requirement MUST be directly traceable to the text.
- If no clear requirements are present, return an empty array [].

For EACH requirement, return a JSON object with:
- id: a temporary string id like "TEMP-001", "TEMP-002", ...
- text: the EXACT requirement text (verbatim if possible)
- type: one of ["functional","non-functional","performance","security","evaluation","legal","compliance","timeline","testing","documentation","integration","hardware","software","communication"]
- section: a short label for the section/category as it appears (or null if unknown)
- confidence: a float 0.0–1.0 (1.0 if exact quote, lower if paraphrase)

Return ONLY a JSON array. No extra text.
"""

def _read_pdf(path: str):
    """Return [(page_no, text)] using PyMuPDF; fallback to pypdf if needed."""
    pages = []
    try:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                text = text.strip()
                logging.info(f"[extract] page {i} chars={len(text)}")
                if text:
                    pages.append((i, text))
    except Exception as e:
        logging.warning(f"PyMuPDF failed: {e}. Falling back to pypdf.")
        from pypdf import PdfReader
        reader = PdfReader(path)
        for i, p in enumerate(reader.pages, start=1):
            text = (p.extract_text() or "").strip()
            logging.info(f"[extract:pypdf] page {i} chars={len(text)}")
            if text:
                pages.append((i, text))
    return pages

def _read_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _token_chunks(text: str, max_tokens: int, overlap: int) -> List[str]:
    ids = enc.encode(text)
    out: List[str] = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        out.append(enc.decode(ids[start:end]))
        if end == len(ids):
            break
        start = max(0, end - overlap)
    return out

def _call_llm_extract(raw: str, page_hint: str | None = None) -> List[Dict[str, Any]]:
    user_prompt = f'Extract requirements from this RFP text{f" ({page_hint})" if page_hint else ""}:\n\n"{raw}"'
    resp = client.chat.completions.create(
        model=EXTRACT_MODEL,
        temperature=0.2,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = resp.choices[0].message.content.strip()
    # Ensure it's valid JSON array
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        # Sometimes the model returns an object with a "requirements" key
        if isinstance(data, dict) and "requirements" in data and isinstance(data["requirements"], list):
            return data["requirements"]
    except Exception:
        # Try to salvage JSON array if extra text slipped in
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group(0))
                if isinstance(arr, list):
                    return arr
            except Exception:
                pass
    return []

def _normalize_type(t: str | None) -> str | None:
    if not t:
        return None
    t = t.lower().strip()
    synonyms = {
        "perf": "performance",
        "security": "security",
        "sec": "security",
        "compliance": "compliance",
        "legal": "legal",
        "functional": "functional",
        "non functional": "non-functional",
        "non-functional": "non-functional",
        "timeline": "timeline",
        "schedule": "timeline",
        "testing": "testing",
        "documentation": "documentation",
        "integration": "integration",
        "hardware": "hardware",
        "software": "software",
        "communication": "communication",
        "evaluation": "evaluation",
    }
    return synonyms.get(t, t)

def _dedupe_requirements(reqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in reqs:
        text = (r.get("text") or "").strip()
        key = re.sub(r"\s+", " ", text.lower())
        if not text or key in seen:
            continue
        seen.add(key)
        # normalize type
        r["type"] = _normalize_type(r.get("type"))
        out.append(r)
    # renumber TEMP ids
    for i, r in enumerate(out, start=1):
        r["id"] = f"TEMP-{i:03d}"
    return out

def extract_requirements_from_file(path: str) -> List[Dict[str, Any]]:
    """
    Reads PDF/DOCX/TXT, chunks if needed, calls LLM per chunk/page,
    merges and deduplicates results. Returns a list of requirement dicts.
    """
    ext = os.path.splitext(path)[1].lower()

    all_results: List[Dict[str, Any]] = []

    if ext == ".pdf":
        pages = _read_pdf(path)
        # Process per-page with a conservative token budget per call
        # (page texts are usually smaller and help traceability)
        for page_no, text in pages:
            # If a single page is large, chunk it
            chunks = _token_chunks(text, MAX_TOKENS_PER_CHUNK, CHUNK_OVERLAP)
            for idx, ch in enumerate(chunks, start=1):
                page_hint = f"Page {page_no}{'' if len(chunks)==1 else f' (part {idx})'}"
                part = _call_llm_extract(ch, page_hint=page_hint)
                # attach page metadata for later UI (optional)
                for p in part:
                    p["section"] = p.get("section") or f"Page {page_no}"
                all_results.extend(part)

    elif ext in (".docx", ".doc"):
        text = _read_docx(path)
        chunks = _token_chunks(text, MAX_TOKENS_PER_CHUNK, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks, start=1):
            part = _call_llm_extract(ch, page_hint=f"Docx part {i}/{len(chunks)}")
            all_results.extend(part)

    elif ext in (".txt",):
        text = _read_txt(path)
        chunks = _token_chunks(text, MAX_TOKENS_PER_CHUNK, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks, start=1):
            part = _call_llm_extract(ch, page_hint=f"Text part {i}/{len(chunks)}")
            all_results.extend(part)

    else:
        # Fallback: try best-effort as plain text
        try:
            text = _read_txt(path)
        except Exception:
            raise ValueError(f"Unsupported file type: {ext}")
        chunks = _token_chunks(text, MAX_TOKENS_PER_CHUNK, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks, start=1):
            part = _call_llm_extract(ch, page_hint=f"File part {i}/{len(chunks)}")
            all_results.extend(part)

    # Merge + dedupe
    return _dedupe_requirements(all_results)
