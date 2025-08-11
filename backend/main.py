from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import tempfile
import os
import shutil   
from dotenv import load_dotenv


load_dotenv()

# --- RAG / LLM utils (to be implemented) ---
from modules.extract import extract_requirements_from_file
#from modules.retrieve import retrieve_from_kb
#from modules.generate import generate_answer_with_refs

app = FastAPI()

# CORS so Lovable can call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, later restrict to your frontend domain
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

import tempfile
import shutil

@app.post("/extract-requirements")
async def extract_requirements(file: UploadFile = File(...), debug: bool = Query(False)):
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Process the file
    reqs = extract_requirements_from_file(tmp_path)

    if debug:
        from modules.extract import _read_pdf
        pages = _read_pdf(tmp_path)
        sample = pages[0][1][:500] if pages else ""
        return {
            "requirements": reqs,
            "debug_sample": sample,
            "pages": len(pages)
        }

    return {"requirements": reqs}



@app.post("/search-docs", response_model=SearchResponse)
async def search_docs(req: Requirement):
    try:
        matches = retrieve_from_kb(req.requirement)
        return {"requirement": req.requirement, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-answer", response_model=GenerateResponse)
async def generate_answer(req: GenerateRequest):
    try:
        answer, refs = generate_answer_with_refs(req.requirement, req.matches)
        return {"answer": answer, "references": refs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"status": "OK", "message": "Coosmo Backend Running"}
