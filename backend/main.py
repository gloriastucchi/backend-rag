from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import tempfile
import os
import shutil
from dotenv import load_dotenv
from supabase import create_client
from modules.supabase_client import get_supabase

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
