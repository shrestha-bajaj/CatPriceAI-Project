from pathlib import Path
from typing import List

DOCS_DIR = Path(__file__).resolve().parent / "docs"


def _load_documents() -> List[dict]:
    docs = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks = []
        current = []
        for line in text.splitlines():
            if line.strip() == "":
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                continue
            current.append(line.strip())
        if current:
            chunks.append(" ".join(current).strip())
        for idx, chunk in enumerate(chunks):
            docs.append({
                "source": path.name,
                "chunk": chunk,
                "index": idx,
            })
    return docs


def _score_chunk(chunk: str, query: str) -> float:
    q_tokens = set(query.lower().split())
    c_tokens = set(chunk.lower().split())
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = q_tokens.intersection(c_tokens)
    return len(overlap) / max(1, len(q_tokens))


def _build_openai_prompt(question: str, chunks: List[str]) -> list[dict]:
    context = "\n\n---\n\n".join(chunks)
    return [
        {
            "role": "system",
            "content": (
                "You are a treaty question-answering assistant. Answer the user's question using only the provided treaty text."
                " Do not invent information or hallucinate. If the answer is not in the provided text, say that you cannot answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Treaty text:\n{context}\n\nQuestion: {question}\n\nAnswer concisely based only on the above treaty text."
            ),
        },
    ]


def _call_openai_chat(api_key: str, question: str, chunks: List[str]) -> str:
    import openai

    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=_build_openai_prompt(question, chunks),
            max_tokens=300,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=_build_openai_prompt(question, chunks),
        max_tokens=300,
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"].strip()


class RagEngine:
    def __init__(self):
        self.documents = _load_documents()

    def answer_question(self, question: str, api_key: str | None = None) -> dict:
        if not self.documents:
            return {"answer": "No treaty documents are available.", "sources": []}

        scored = [(doc, _score_chunk(doc["chunk"], question)) for doc in self.documents]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = [item for item in scored if item[1] > 0]
        if not top:
            return {"answer": "I could not find the answer in the provided documents.", "sources": []}

        top_chunks = [doc["chunk"] for doc, _ in top[:3]]
        sources = sorted({doc["source"] for doc, _ in top[:3]})

        if api_key:
            try:
                answer = _call_openai_chat(api_key, question, top_chunks)
                return {"answer": answer, "sources": sources}
            except Exception as exc:
                answer = "\n\n".join(top_chunks[:3])
                return {
                    "answer": f"{answer}\n\n(Note: OpenAI RAG fallback due to: {exc})",
                    "sources": sources,
                }

        answer = "\n\n".join(top_chunks[:3])
        return {"answer": answer, "sources": sources}
