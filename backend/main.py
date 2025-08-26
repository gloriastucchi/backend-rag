from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import tempfile
from modules.auth import get_user_id
import os
import shutil
from dotenv import load_dotenv
from supabase import create_client
from modules.supabase_client import get_supabase
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from modules.supabase_client import get_supabase, get_service_client  # come hai già fatto
from modules.embed import embed_texts  # wrapper openai embeddings
import tempfile, os, shutil
from uuid import uuid4
from fastapi import Depends, Body
from uuid import uuid4
from modules.supabase_client import get_service_client
from modules.embed import embed_and_upsert_document_text, delete_document_chunks
import os


load_dotenv()

# --- RAG / LLM utils ---
from modules.extract import extract_requirements_from_file
# from modules.retrieve import retrieve_from_kb
# from modules.generate import generate_answer_with_refs

# --- Supabase helper ---
from modules.supabase_client import get_supabase

app = FastAPI()

# CORS per dev (restringi in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_user_id() -> str:
    """
    DEV: prende l'utente da ENV `DEV_USER_ID`.
    In prod sostituisci con estrazione dal JWT.
    """
    uid = os.getenv("DEV_USER_ID")
    if not uid:
        raise HTTPException(401, "Missing user id (set DEV_USER_ID or wire JWT auth)")
    return uid

KB_BUCKET = os.getenv("KB_BUCKET", "project-kb")

def _extract_text_from_bytes(data: bytes, filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(p.get_text("text") for p in doc)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ---------- MODELS ----------
class Requirement(BaseModel):
    requirement: str

class SearchResponse(BaseModel):
    requirement: str
    matches: List[Dict[str, Any]]

class GenerateRequest(BaseModel):
    requirement: str
    matches: List[Dict[str, Any]]

class GenerateResponse(BaseModel):
    answer: str
    references: List[str]

# ---------- ENDPOINTS ----------

import traceback

@app.post("/extract-requirements")
async def extract_requirements(
    file: UploadFile = File(...),
    debug: bool = Query(False),
    save: bool = Query(False),
    project_id: str | None = Query(default=None),
    title: str | None = Query(default=None),
):
    tmp_path = None
    try:
        # 1) save the PDF to /tmp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # 2) extract requirements
        reqs = extract_requirements_from_file(tmp_path)
        result: Dict[str, Any] = {"requirements": reqs}

        if save:
            if not project_id:
                raise HTTPException(status_code=400, detail="project_id required when save=true")

            sb = get_supabase()

            # 3a) create document row with an allowed status
            payload = {
                "project_id": project_id,
                "title": title or file.filename,
                "file_name": file.filename,
                "status": "uploaded",          # ✅ allowed by your CHECK
                # "user_id": <add later when you have auth/JWT>,
                # "user_email": <optional mirror>
            }
            doc_ins = sb.table("rfq_documents").insert(payload).execute()
            if not doc_ins or not getattr(doc_ins, "data", None):
                raise HTTPException(status_code=500, detail="Failed to insert rfq_documents")
            doc_id = doc_ins.data[0].get("id")

            # 3b) (optional) mark as analyzing
            sb.table("rfq_documents").update({"status": "analyzing"}).eq("id", doc_id).execute()

            # 3c) bulk insert requirements
            rows = [{
                "document_id": doc_id,
                "req_id": r.get("id"),
                "text": r.get("text"),
                "type": r.get("type"),
                "section": r.get("section"),
                "confidence": r.get("confidence"),
            } for r in (reqs or [])]

            if rows:
                sb.table("rfq_requirements").insert(rows).execute()

            # 3d) finally mark as analyzed
            sb.table("rfq_documents").update({"status": "analyzed"}).eq("id", doc_id).execute()

            result["document_id"] = doc_id

        return result

    except Exception as e:
        print("\n--- ERROR in /extract-requirements ---")
        print(str(e))
        traceback.print_exc()
        print("--- END ERROR ---\n")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/kb/upload")
async def kb_upload(
    file: UploadFile = File(...),
    user_id: str = Depends(get_user_id),
):
    # usa SEMPRE il service client nel backend per storage + DB write
    sb = get_service_client()
    storage = sb.storage

    # 1) Storage upload: userId/<uuid>_<original>
    storage_path = f"{user_id}/{uuid4()}_{file.filename}"
    file_bytes = await file.read()
    try:
        storage.from_(KB_BUCKET).upload(storage_path, file_bytes)
    except Exception as e:
        raise HTTPException(500, f"Storage upload failed: {e}")

    # 2) Insert in user_documents (schema attuale: file_name/file_path)
    payload = {
        "user_id": user_id,
        "file_name": file.filename,
        "file_path": storage_path,
        "checksum": None,
    }
    ins = sb.table("user_documents").insert(payload).select("id, file_name").single().execute()
    if not ins or not getattr(ins, "data", None):
        raise HTTPException(500, "DB insert failed for user_documents")
    document_id = ins.data["id"]
    filename = ins.data.get("file_name") or file.filename

    # 3) Estrai testo e crea embedding
    text = _extract_text_from_bytes(file_bytes, filename)
    if not text.strip():
        raise HTTPException(400, "No extractable text")

    delete_document_chunks(document_id)  # idempotenza
    summary = embed_and_upsert_document_text(
        text=text,
        user_id=user_id,
        document_id=document_id,
    )

    return {"document_id": document_id, "embedded_chunks": summary["chunks"]}

class EmbedOneIn(BaseModel):
    document_id: str  # uuid

@app.post("/kb/embed-one")
def kb_embed_one(body: EmbedOneIn, user_id: str = Depends(get_user_id)):
    sb = get_supabase()
    svc = get_service_client()

    doc = sb.table("user_documents").select("*").eq("id", body.document_id).single().execute().data
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc["user_id"] != user_id:
        raise HTTPException(403, "Forbidden")

    filename = doc.get("filename") or doc.get("file_name")
    path = doc.get("storage_path") or doc.get("file_path")
    if not filename or not path:
        raise HTTPException(400, "Missing filename/storage_path")

    file_bytes = svc.storage.from_(KB_BUCKET).download(path)
    if file_bytes is None:
        raise HTTPException(500, "Storage download failed")

    delete_document_chunks(body.document_id)
    text = _extract_text_from_bytes(file_bytes, filename)
    if not text.strip():
        raise HTTPException(400, "No extractable text")

    summary = embed_and_upsert_document_text(text=text, user_id=user_id, document_id=body.document_id)
    return {"document_id": body.document_id, "embedded_chunks": summary["chunks"]}

class SearchIn(BaseModel):
    query: str

@app.post("/kb/search")
def kb_search(payload: SearchIn, user_id: str = Depends(get_user_id)):
    from openai import OpenAI
    import psycopg2, psycopg2.extras

    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    qvec = OpenAI().embeddings.create(model=EMBED_MODEL, input=payload.query).data[0].embedding

    pg_url = os.getenv("SUPABASE_DB_CONNECTION_STRING")  # postgres://user:pass@host:port/db
    if not pg_url:
        raise HTTPException(500, "Missing SUPABASE_DB_CONNECTION_STRING")

    sql = """
    select
      ue.document_id,
      ue.chunk_index,
      ue.content,
      (ue.embedding <-> %s) as distance,
      ud.filename, ud.file_name, ud.storage_path, ud.file_path, ud.created_at
    from user_embeddings ue
    join user_documents ud on ud.id = ue.document_id
    where ue.user_id = %s
    order by ue.embedding <-> %s
    limit 10;
    """

    try:
        conn = psycopg2.connect(pg_url)
        with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (qvec, user_id, qvec))
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(500, f"Search failed: {e}")

    # normalizza campi per il frontend
    for r in rows:
        r["filename"] = r.get("filename") or r.get("file_name")
        r["storage_path"] = r.get("storage_path") or r.get("file_path")

    return {"results": rows}



@app.post("/kb/process")
async def kb_process(doc_id: str, user_id: str = Depends(get_user_id)):
    sb = get_supabase()
    svc = get_service_client()

    # 0) fetch doc
    doc = sb.table("kb_documents").select("*").eq("id", doc_id).single().execute().data
    if not doc or doc["user_id"] != user_id:
        raise HTTPException(404, "Document not found")

    # 1) download
    raw = svc.storage.from_("kb_documents").download(doc["file_path"])
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    # 2) estrai testo + chunking (usa il tuo parser PDF)
    from modules.text import extract_text, chunk_text
    text = extract_text(tmp_path)
    chunks = chunk_text(text, size=900, overlap=150)

    # 3) embeddings
    embs = embed_texts([c["content"] for c in chunks])  # -> list[list[float]]

    rows = []
    for i, c in enumerate(chunks):
        rows.append({
            "user_id": user_id,
            "project_id": doc["project_id"],
            "doc_id": doc_id,
            "chunk_idx": i,
            "content": c["content"],
            "embedding": embs[i],
        })

    if rows:
        sb.table("kb_embeddings").insert(rows).execute()
        sb.table("kb_documents").update({"status": "ready"}).eq("id", doc_id).execute()
    os.remove(tmp_path)
    return {"chunks": len(rows), "status": "ready"}

@app.post("/extract-requirements-from-doc")
async def extract_requirements_from_existing_doc(
    document_id: str,
    save: bool = True
):
    """
    Use when the PDF is already uploaded+tracked in rfq_documents.
    We fetch it from Supabase Storage (bucket rfq_documents), run extraction,
    and write results to rfq_requirements.
    """

    sb = get_supabase()

    # 0) fetch rfq_documents row
    res = sb.table("rfq_documents").select("*").eq("id", document_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="document not found")

    doc = res.data
    file_path = doc.get("file_path")   # e.g. "user123/abc/testRFP.pdf"
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path missing on rfq_documents")

    bucket = "rfq_documents"

    # 1) download file temporarily
    client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
    
    try:
        file_bytes = client.storage.from_(bucket).download(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to download: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # 2) extract requirements
    reqs = extract_requirements_from_file(tmp_path)

    # delete tmp
    os.remove(tmp_path)

    # 3) optionally save to rfq_requirements
    if save and reqs:
        rows = [{
            "document_id": document_id,
            "req_id": r.get("id"),
            "text": r.get("text"),
            "type": r.get("type"),
            "section": r.get("section"),
            "confidence": r.get("confidence"),
        } for r in reqs]
        sb.table("rfq_requirements").insert(rows).execute()

    return {
        "document_id": document_id,
        "requirements": reqs
    }


@app.post("/search-docs", response_model=SearchResponse)
async def search_docs(req: Requirement):
    try:
        matches = retrieve_from_kb(req.requirement)  # da implementare
        return {"requirement": req.requirement, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-answer", response_model=GenerateResponse)
async def generate_answer(req: GenerateRequest):
    try:
        answer, refs = generate_answer_with_refs(req.requirement, req.matches)  # da implementare
        return {"answer": answer, "references": refs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"status": "OK", "message": "Coosmo Backend Running"}
