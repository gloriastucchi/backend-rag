from typing import Optional, Dict, Any
from supabase import create_client, Client
from .settings import SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, SUPABASE_TABLE

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def storage_upload(path: str, content: bytes, content_type: str = "application/octet-stream"):
    return sb.storage.from_(SUPABASE_BUCKET).upload(path, content, {
        "contentType": content_type,
        "upsert": True
    })

def table_insert(row: Dict[str, Any], table: Optional[str] = None):
    return sb.table(table or SUPABASE_TABLE).insert(row).execute()

def table_bulk_insert(rows, table: Optional[str] = None):
    return sb.table(table or SUPABASE_TABLE).insert(rows).execute()

def rpc(name: str, params: dict):
    return sb.rpc(name, params).execute()
