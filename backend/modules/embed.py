# modules/embed.py
from __future__ import annotations

import os
import time
import uuid
from typing import Iterable, List, Dict, Any, Optional

from openai import OpenAI
from supabase import create_client

# helper Supabase definito usare:
# from modules.supabase_client import get_supabase

# -------------------------
# Config
# -------------------------
OPENAI_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536 dims
CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE", "800"))       # ~ parole/caratteri: heuristica
CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP", "200")) # overlap tra chunk
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "96"))
TABLE_NAME = os.getenv("EMBED_TABLE", "kb_chunks")

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
# Token / chunk util
# -------------------------
def rough_token_count(text: str) -> int:
    """Stima grezza token, ok per logging/QA."""
    # sostituire con tiktoken x maggiore precisione
    return max(1, int(len(text.split()) * 1.3))

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Chunk ingenuo per paragrafi/righe con overlap a livello di caratteri.
    Puoi sostituirlo con un chunker più evoluto (per parole/sentenze).
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    i = 0
    n = len(text)
    step = max(1, chunk_size - overlap)

    while i < n:
        end = min(n, i + chunk_size)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i += step

    return chunks

# -------------------------
# OpenAI embeddings
# -------------------------
def embed_texts(
    texts: List[str],
    model: str = OPENAI_EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
    max_retries: int = 3,
) -> List[List[float]]:
    """
    Esegue embedding batched per lista di testi.
    Restituisce una lista di vettori nello stesso ordine.
    """
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
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                # backoff semplice
                time.sleep(1.5 * attempt)
    return vectors

def get_embedding(text: str, model: str = OPENAI_EMBED_MODEL) -> List[float]:
    """Shortcut per singolo testo."""
    resp = _openai.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding

# -------------------------
# Upsert in Supabase
# -------------------------
def upsert_chunks(
    *,
    chunks: List[str],
    embeddings: List[List[float]],
    user_id: str,
    document_id: str,
    project_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Inserisce/aggiorna i chunk nella tabella vettoriale.
    Usa upsert con chiave (document_id, chunk_index) se in DB hai vincolo unico.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have same length")

    rows = []
    for idx, (content, vec) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "project_id": project_id,
            "document_id": document_id,
            "chunk_index": idx,
            "content": content,
            "embedding": vec,           # per vector est, passare la lista
            "tokens": rough_token_count(content),
        })

    sb = _supabase()
    # con vincolo unico su (document_id, chunk_index) usare on_conflict:
    # res = sb.table(TABLE_NAME).upsert(rows, on_conflict="document_id,chunk_index").execute()
    res = sb.table(TABLE_NAME).upsert(rows).execute()
    return res.data or []

def delete_document_chunks(document_id: str) -> int:
    """Cancella tutti i chunk relativi a un documento."""
    sb = _supabase()
    res = sb.table(TABLE_NAME).delete().eq("document_id", document_id).execute()
    return len(res.data or [])

# -------------------------
# Pipeline principale
# -------------------------
def embed_and_upsert_document_text(
    *,
    text: str,
    user_id: str,
    document_id: str,
    project_id: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    model: str = OPENAI_EMBED_MODEL,
) -> Dict[str, Any]:
    """
    1) fa chunk del testo
    2) embed in batch
    3) upsert su Supabase
    4) ritorna summary
    """
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return {"document_id": document_id, "chunks": 0, "inserted": 0}

    vectors = embed_texts(chunks, model=model)
    inserted = upsert_chunks(
        chunks=chunks,
        embeddings=vectors,
        user_id=user_id,
        document_id=document_id,
        project_id=project_id,
    )
    return {
        "document_id": document_id,
        "chunks": len(chunks),
        "inserted": len(inserted),
        "model": model,
    }
