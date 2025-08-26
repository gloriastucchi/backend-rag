# modules/embed.py
from __future__ import annotations

import os
import time
from typing import List, Dict, Any

from openai import OpenAI
from supabase import create_client

# -------------------------
# Config
# -------------------------
OPENAI_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536 dims
CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE", "1800"))
CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "96"))
TABLE_NAME = os.getenv("EMBED_TABLE", "user_embeddings")  # <-- allineato

# -------------------------
# Clients
# -------------------------
_openai = OpenAI()

def _supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # server-side ONLY
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars.")
    return create_client(url, key)

# -------------------------
# Chunk utils (caratteri con overlap)
# -------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    i, n = 0, len(text)
    step = max(1, chunk_size - overlap)
    while i < n:
        j = min(n, i + chunk_size)
        c = text[i:j].strip()
        if c:
            chunks.append(c)
        if j == n:
            break
        i += step
    return chunks

# -------------------------
# OpenAI embeddings
# -------------------------
def embed_texts(texts: List[str], model: str = OPENAI_EMBED_MODEL, batch_size: int = BATCH_SIZE, max_retries: int = 3) -> List[List[float]]:
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        attempt = 0
        while True:
            try:
                resp = _openai.embeddings.create(model=model, input=batch)
                for d in resp.data:
                    vectors.append(d.embedding)
                break
            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(1.5 * attempt)
    return vectors

# -------------------------
# DB ops
# -------------------------
def upsert_chunks(*, chunks: List[str], embeddings: List[List[float]], user_id: str, document_id: str) -> List[Dict[str, Any]]:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have same length")

    rows = []
    for idx, (content, vec) in enumerate(zip(chunks, embeddings)):
        rows.append({
            # id: NON lo mettiamo: è bigserial e lo genera Postgres
            "user_id": user_id,
            "document_id": document_id,
            "chunk_index": idx,
            "content": content,
            "embedding": vec,  # Supabase accetta la lista per pgvector
        })

    sb = _supabase()
    # Se hai creato una unique (document_id, chunk_index), puoi usare on_conflict
    res = sb.table(TABLE_NAME).upsert(rows, on_conflict="document_id,chunk_index").execute()
    return res.data or []

def delete_document_chunks(document_id: str) -> int:
    sb = _supabase()
    res = sb.table(TABLE_NAME).delete().eq("document_id", document_id).execute()
    return len(res.data or [])

# -------------------------
# Pipeline
# -------------------------
def embed_and_upsert_document_text(*, text: str, user_id: str, document_id: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, model: str = OPENAI_EMBED_MODEL) -> Dict[str, Any]:
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return {"document_id": document_id, "chunks": 0, "inserted": 0}

    vectors = embed_texts(chunks, model=model)
    inserted = upsert_chunks(chunks=chunks, embeddings=vectors, user_id=user_id, document_id=document_id)
    return {"document_id": document_id, "chunks": len(chunks), "inserted": len(inserted), "model": model}
