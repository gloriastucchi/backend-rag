from openai import OpenAI
from supabase import create_client
from .settings import SUPABASE_URL, SUPABASE_KEY, MATCH_FN, EMBED_MODEL, CHAT_MODEL, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

def embed(text: str):
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def search(query: str, match_count=5, threshold=0.2):
    q_emb = embed(query)
    rpc = sb.rpc(MATCH_FN, {
        "query_embedding": q_emb,
        "match_count": match_count,
        "match_threshold": threshold
    }).execute()
    return rpc.data or []

def answer_with_context(query: str, matches: list[str]):
    context = "\n\n".join([m.get("content", "") for m in matches if m.get("content")])
    chat = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.2,
        messages=[
            {"role": "system", "content": "Answer using only the provided context. If unsure, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return chat.choices[0].message.content.strip()
