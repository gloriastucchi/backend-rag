from supabase import create_client
from .settings import SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_to_storage(file_bytes: bytes, path: str, content_type: str):
    return sb.storage.from_(SUPABASE_BUCKET).upload(
        path, file_bytes, {"contentType": content_type, "upsert": True}
    )
