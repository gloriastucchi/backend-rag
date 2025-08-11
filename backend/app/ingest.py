from typing import List, Tuple, Dict
from io import BytesIO
from pypdf import PdfReader
import tiktoken
from openai import OpenAI
from .settings import OPENAI_API_KEY, EMBED_MODEL, CHUNK_TOKENS, CHUNK_OVERLAP
from .supa import storage_upload

client = OpenAI(api_key=OPENAI_API_KEY)
enc = tiktoken.get_encoding("cl100k_base")

def extract_text_from_pdf(binary: bytes) -> List[Tuple[int, str]]:
    pages = []
    reader = PdfReader(BytesIO(binary))
    for i, p in enumerate(reader.pages, 1):
        text = (p.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages

def chunk(text: str, max_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP) -> List[str]:
    ids = enc.encode(text)
    out, start = [], 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        out.append(enc.decode(ids[start:end]))
        if end == len(ids): break
        start = max(0, end - overlap)
    return out

def ingest_pdf(file_bytes: bytes, filename: str, meta: Dict) -> Dict:
    # 1) store original file (if Supabase configured)
    storage_path = f"rfq/{filename}"
    storage_upload(storage_path, file_bytes, "application/pdf")

    # 2) parse + chunk (embedding happens later when DB is ready)
    pages = extract_text_from_pdf(file_bytes)
    total_chunks = sum(len(chunk(t)) for _, t in pages)

    return {
        "docId": storage_path,
        "title": filename.rsplit(".", 1)[0],
        "pages": len(pages),
        "chunks": total_chunks,
        "note": "stored & parsed; embedding will activate when DB is configured"
    }
