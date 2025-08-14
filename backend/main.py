from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import tempfile
import os
import shutil
from dotenv import load_dotenv

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
        # 1) salva il PDF su /tmp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # 2) estrai i requisiti
        reqs = extract_requirements_from_file(tmp_path)
        result: Dict[str, Any] = {"requirements": reqs}

        # 3) opzionale: persisti su Supabase
        if save:
            if not project_id:
                raise HTTPException(status_code=400, detail="project_id required when save=true")

            sb = get_supabase()

            # 3a) crea record documento
            doc_ins = (
                sb.table("rfq_documents")
                  .insert({
                      "project_id": project_id,
                      "title": title or file.filename,
                      "status": "extracted",
                      "file_name": file.filename,
                  })
                  .select("id")
                  .single()
                  .execute()
            )
            doc_id = doc_ins.data["id"]

            # 3b) bulk insert requisiti
            rows = [{
                "document_id": doc_id,
                "req_id": r.get("id"),
                "text": r.get("text"),
                "type": r.get("type"),
                "section": r.get("section"),
                "confidence": r.get("confidence"),
            } for r in reqs]

            if rows:
                sb.table("rfq_requirements").insert(rows).execute()

            result["document_id"] = doc_id

        # 4) opzionale: payload di debug
        if debug:
            from modules.extract import _read_pdf
            pages = _read_pdf(tmp_path)
            sample = pages[0][1][:500] if pages else ""
            result.update({"debug_sample": sample, "pages": len(pages)})

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
