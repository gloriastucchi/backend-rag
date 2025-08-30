from openai import OpenAI

client = OpenAI()

def extract_scope_and_deliverables_from_file(file_path: str):
    # 1. Carica il testo del file (qui placeholder, meglio usare parser pdf)
    with open(file_path, "r", errors="ignore") as f:
        content = f.read()

    # 2. Prompt al modello
    prompt = f"""
    Extract two sections from the following RFQ text:

    1. Scope of Work: provide a clean narrative summary (can include bullet points).
    2. Deliverables: provide a bullet point list.

    RFQ content:
    {content[:4000]}   # taglia per non esplodere il token limit
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    text = response.choices[0].message.content

    # 3. Split sections
    scope_text = ""
    deliverables = []

    if "Deliverables:" in text:
        parts = text.split("Deliverables:")
        scope_text = parts[0].replace("Scope of Work:", "").strip()
        deliverables = [line.strip(" -•*") for line in parts[1].split("\n") if line.strip()]
    else:
        scope_text = text.strip()

    return scope_text, deliverables
