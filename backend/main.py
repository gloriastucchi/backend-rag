from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
from uuid import uuid4
import os
import shutil
import tempfile
import asyncio
import traceback

from supabase import create_client  # usato per download dallo Storage
from modules.scope_extraction import extract_scope_and_deliverables_from_file
from modules.supabase_client import get_supabase, get_service_client
from modules.embed import embed_and_upsert_document_text, delete_document_chunks
from modules.extract import extract_requirements_from_file

load_dotenv()

app = FastAPI()

# --------- ENV ---------
KB_BUCKET = os.getenv("KB_BUCKET", "project-kb")
RFQ_BUCKET = os.getenv("RFQ_BUCKET", "rfq_documents")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# --------- CORS (restringi in prod) ---------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- UTILS ---------
def _log(step: str, **kv):
    print(f"[extract-scope] {step} :: " + " ".join(f"{k}={kv[k]}" for k in kv))

async def get_user_id(user_id_header: str | None = Header(None, alias="X-User-Id")) -> str:
    if not user_id_header:
        raise HTTPException(status_code=422, detail="Missing X-User-Id header")
    return user_id_header

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

class SearchIn(BaseModel):
    query: str

# ---------- ENDPOINTS ----------

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
                "status": "uploaded",
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


@app.post("/extract-scope-from-doc")
async def extract_scope_from_existing_doc(
    document_id: str = Query(..., description="UUID rfq_documents.id"),
    save: bool = Query(True),
    dry_run: bool = Query(False, description="Se true, non chiama l'LLM (debug veloce)"),
):
    """
    Estrae:
      - scope_of_work_text: string
      - deliverables: list[str]

    Se save=True:
      - salva scope_of_work_text in rfq_documents.analysis_data
      - salva deliverables in rfq_scope_items (section='Deliverables')
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(500, detail="Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

    sb = get_supabase()
    tmp_path = None
    try:
        _log("START", document_id=document_id, save=save, dry_run=dry_run)

        # 0) fetch rfq_documents row
        res = sb.table("rfq_documents").select("*").eq("id", document_id).single().execute()
        if not res or not getattr(res, "data", None):
            _log("DOC_NOT_FOUND")
            raise HTTPException(status_code=404, detail="document not found")

        doc = res.data
        file_path = doc.get("file_path")
        file_name = doc.get("file_name") or "document.pdf"
        if not file_path:
            _log("MISSING_FILE_PATH")
            raise HTTPException(status_code=400, detail="file_path missing on rfq_documents")

        _log("DOC_OK", file_path=file_path, file_name=file_name)

        # 1) download file temporarily
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        _log("DOWNLOAD_BEGIN")
        try:
            file_bytes = client.storage.from_(RFQ_BUCKET).download(file_path)
        except Exception as e:
            _log("DOWNLOAD_ERROR", err=str(e))
            raise HTTPException(status_code=500, detail=f"Unable to download: {e}")
        _log("DOWNLOAD_OK", size=len(file_bytes or b""))

        suffix = os.path.splitext(file_name)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        _log("TMP_READY", tmp_path=tmp_path)

        # 2) extraction (LLM) con timeout e/o dry-run
        if dry_run:
            _log("DRY_RUN")
            scope_text = "This is a DRY-RUN scope text (no LLM)."
            deliverables = ["Item A", "Item B", "Item C"]
        else:
            async def _run():
                return extract_scope_and_deliverables_from_file(tmp_path)

            _log("LLM_BEGIN")
            try:
                out = await asyncio.wait_for(_run(), timeout=90)  # timeout duro
            except asyncio.TimeoutError:
                _log("LLM_TIMEOUT")
                raise HTTPException(504, detail="Scope extraction timed out")
            except Exception as e:
                _log("LLM_ERROR", err=str(e))
                traceback.print_exc()
                raise HTTPException(500, detail=f"Scope extraction failed: {e}")

            scope_text = (out or {}).get("scope_of_work_text") or ""
            deliverables = (out or {}).get("deliverables") or []
            _log("LLM_OK", scope_len=len(scope_text), deliverables=len(deliverables))

        # 3) save (optional)
        if save:
            _log("SAVE_BEGIN")
            sb.table("rfq_documents").update({"status": "analyzing"}).eq("id", document_id).execute()

            # idempotenza deliverables
            sb.table("rfq_scope_items").delete().eq("document_id", document_id).eq("section", "Deliverables").execute()
            if deliverables:
                rows = [{
                    "document_id": document_id,
                    "section": "Deliverables",
                    "text": d,
                    "order_index": i,
                    "confidence": 0.9,
                } for i, d in enumerate(deliverables)]
                sb.table("rfq_scope_items").insert(rows).execute()

            prev = doc.get("analysis_data") or {}
            prev["scope_of_work_text"] = scope_text
            prev["deliverables_count"] = len(deliverables)

            sb.table("rfq_documents").update({
                "status": "analyzed",
                "analysis_data": prev
            }).eq("id", document_id).execute()
            _log("SAVE_OK")

        _log("END_OK")
        return {
            "document_id": document_id,
            "scope_of_work_text": scope_text,
            "deliverables": deliverables
        }

    except HTTPException:
        raise
    except Exception as e:
        _log("FATAL", err=str(e))
        traceback.print_exc()
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
    sb = get_service_client()
    storage = sb.storage

    # 1) Storage upload: userId/<uuid>_<original>
    storage_path = f"{user_id}/{uuid4()}_{file.filename}"
    file_bytes = await file.read()
    try:
        storage.from_(KB_BUCKET).upload(storage_path, file_bytes)
    except Exception as e:
        raise HTTPException(500, f"Storage upload failed: {e}")

    # 2) Insert in user_documents
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

    return {
        "document_id": document_id,
        "chunks": summary["chunks"],
        "inserted": summary["inserted"],
        "model": summary["model"],
        "debug": {"text_len": len(text), "first_120": text[:120]},
    }


class EmbedOneIn(BaseModel):
    document_id: str  # uuid

@app.post("/kb/embed-one")
def kb_embed_one(body: EmbedOneIn, user_id: str = Depends(get_user_id)):
    sb = get_service_client()
    storage = sb.storage

    doc = sb.table("user_documents").select("*").eq("id", body.document_id).single().execute().data
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc["user_id"] != user_id:
        raise HTTPException(403, "Forbidden")

    filename = doc.get("file_name") or doc.get("filename")
    path = doc.get("file_path") or doc.get("storage_path")
    if not filename or not path:
        raise HTTPException(400, "Missing filename/storage_path")

    try:
        file_bytes = storage.from_(KB_BUCKET).download(path)
    except Exception as e:
        raise HTTPException(500, f"Storage download failed: {e}")

    delete_document_chunks(body.document_id)
    text = _extract_text_from_bytes(file_bytes, filename)
    if not text.strip():
        raise HTTPException(400, "No extractable text")

    summary = embed_and_upsert_document_text(
        text=text,
        user_id=user_id,
        document_id=body.document_id
    )
    return {"document_id": body.document_id, "embedded_chunks": summary["chunks"]}


# --- RAG demo search (ok lasciare così se non usi) ---
from openai import OpenAI
import psycopg2, psycopg2.extras
from datetime import timedelta
client = OpenAI()

TOP_K = int(os.getenv("RAG_TOP_K", "8"))

def _pg():
    url = os.getenv("SUPABASE_DB_CONNECTION_STRING")
    if not url:
        raise HTTPException(500, "Missing SUPABASE_DB_CONNECTION_STRING")
    return psycopg2.connect(url)

@app.post("/kb/search")
def kb_search(payload: SearchIn, user_id: str = Depends(get_user_id)):
    qvec = client.embeddings.create(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        input=payload.query
    ).data[0].embedding

    sql = """
    select
      ue.document_id,
      ue.chunk_index,
      ue.content,
      (ue.embedding <-> %s) as distance,
      ud.filename,
      ud.storage_path,
      ud.created_at
    from user_embeddings ue
    join user_documents ud on ud.id = ue.document_id
    where ue.user_id = %s
    order by ue.embedding <-> %s
    limit %s;
    """

    conn = _pg()
    with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (qvec, user_id, qvec, TOP_K))
        rows = cur.fetchall()
    conn.close()

    svc = get_service_client()
    out = []
    for r in rows:
        url = svc.storage.from_(KB_BUCKET).create_signed_url(
            r["storage_path"], int(timedelta(hours=2).total_seconds())
        )["signedURL"]
        out.append({**r, "signed_url": url})
    return {"results": out}


@app.post("/kb/process")
async def kb_process(doc_id: str, user_id: str = Depends(get_user_id)):
    sb = get_service_client()
    storage = sb.storage

    doc = sb.table("user_documents").select("*").eq("id", doc_id).single().execute().data
    if not doc or doc["user_id"] != user_id:
        raise HTTPException(404, "Document not found")

    filename = doc.get("file_name") or doc.get("filename")
    path = doc.get("file_path") or doc.get("storage_path")
    if not filename or not path:
        raise HTTPException(400, "Missing filename/storage_path")

    try:
        file_bytes = storage.from_(KB_BUCKET).download(path)
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")

    text = _extract_text_from_bytes(file_bytes, filename)
    if not text.strip():
        raise HTTPException(400, "No extractable text")

    delete_document_chunks(doc_id)
    summary = embed_and_upsert_document_text(
        text=text,
        user_id=user_id,
        document_id=doc_id
    )
    return {"document_id": doc_id, "embedded_chunks": summary["chunks"], "status": "ready"}


@app.post("/extract-requirements-from-doc")
async def extract_requirements_from_existing_doc(
    document_id: str,
    save: bool = True
):
    sb = get_supabase()

    res = sb.table("rfq_documents").select("*").eq("id", document_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="document not found")

    doc = res.data
    file_path = doc.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path missing on rfq_documents")

    bucket = RFQ_BUCKET

    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    try:
        file_bytes = client.storage.from_(bucket).download(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to download: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        reqs = extract_requirements_from_file(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

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

    return {"document_id": document_id, "requirements": reqs}


# ---- Scope: lettura risultati salvati ----
@app.get("/rfq/scope")
def get_scope(document_id: str):
    print(f"[GET /rfq/scope] document_id={document_id}")
    sb = get_supabase()
    dres = sb.table("rfq_documents").select("analysis_data").eq("id", document_id).single().execute()
    if not dres.data:
        print(f"[GET /rfq/scope] NOT FOUND: {document_id}")
        raise HTTPException(status_code=404, detail="document not found")

    analysis = dres.data.get("analysis_data") or {}
    scope_text = analysis.get("scope_of_work_text") or ""

    ires = (sb.table("rfq_scope_items")
              .select("text, order_index")
              .eq("document_id", document_id)
              .eq("section", "Deliverables")
              .order("order_index", desc=False)
              .execute())
    deliverables = [row["text"] for row in (ires.data or [])]
    print(f"[GET /rfq/scope] deliverables={len(deliverables)} chars(scope)={len(scope_text)}")

    return {
        "document_id": document_id,
        "scope_of_work_text": scope_text,
        "deliverables": deliverables
    }


@app.get("/")
async def root():
    return {"status": "OK", "message": "Coosmo Backend Running"}
