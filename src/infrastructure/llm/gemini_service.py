import os
import re
import time
from pathlib import Path
from typing import List

import chromadb
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

from src.domain.entities import ChatMessage
from src.interfaces.gateways import LlmServiceGateway

CHAT_MODEL = "gemini-flash-latest"
EMBED_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "flask_code"
TOP_K = 5

RAG_SYSTEM_PROMPT = """You are a **Flask source code expert**. Answer naturally. Do not announce that you are using context or snippets. Just answer directly.

## Grounding
- Your knowledge is limited to the Context below. Never use pre-training knowledge.
- If the Context lacks the answer, say: "I don't have enough source code coverage to answer that."
- Cite file paths and function names inline: in `src/flask/app.py`, `Flask.dispatch_request`.
- Do NOT add any "I found in the context" or "based on the snippets" language. Just state the facts.

## Formatting
- Use proper **Markdown**: fenced code blocks (` ```python `), inline code, bold, lists.
- Use **inline code** for functions, classes, paths, variables.

## Exclusions
- Never invent APIs or signatures not in the snippets.
- Never use external libraries, extensions, or third-party tools.
- Never give deployment, security, or production advice.
- Never mention licensing, pricing, or version history."""


class GeminiRagService(LlmServiceGateway):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        self.client = genai.Client(api_key=api_key)

        chroma_path = str(Path(__file__).resolve().parents[3] / "chroma_db")
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = chroma_client.get_collection(COLLECTION_NAME)

    def _embed_texts(self, texts, task_type="RETRIEVAL_QUERY", batch_size=20, max_retries=6):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            attempt = 0
            while True:
                try:
                    result = self.client.models.embed_content(
                        model=EMBED_MODEL,
                        contents=batch,
                        config=genai_types.EmbedContentConfig(task_type=task_type),
                    )
                    break
                except genai_errors.ClientError as e:
                    if "429" in str(e):
                        attempt += 1
                        if attempt > max_retries:
                            raise
                        wait = min(2**attempt, 30)
                        time.sleep(wait)
                    else:
                        raise
            all_embeddings.extend([e.values for e in result.embeddings])
            time.sleep(0.5)
        return all_embeddings

    def retrieve(self, query, top_k=TOP_K):
        query_embedding = self._embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "text": doc,
                "file": meta["file"],
                "name": meta["name"],
                "type": meta["type"],
                "distance": dist,
            })
        return retrieved

    def chat_follow_up(self, history: List[ChatMessage], new_message: str) -> str:
        return self.generate(new_message, history)

    def generate(self, message: str, history: List[ChatMessage]) -> str:
        retrieved = self.retrieve(message, top_k=TOP_K)

        context_block = "\n\n".join(
            f"[{r['file']} :: {r['name']}]\n{r['text']}" for r in retrieved
        )

        history_block = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in history[-6:]
        )

        full_prompt = f"""{RAG_SYSTEM_PROMPT}

Conversation so far:
{history_block}

Context:
{context_block}

Question: {message}

Answer:"""

        try:
            full_text = ""
            for chunk in self.client.models.generate_content_stream(
                model=CHAT_MODEL, contents=full_prompt
            ):
                if chunk.text:
                    full_text += chunk.text
        except genai_errors.ClientError as e:
            raise RuntimeError(self._friendly_error(e)) from e

        return full_text.strip()

    def generate_stream(self, message: str, history: List[ChatMessage]):
        retrieved = self.retrieve(message, top_k=TOP_K)

        context_block = "\n\n".join(
            f"[{r['file']} :: {r['name']}]\n{r['text']}" for r in retrieved
        )

        history_block = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in history[-6:]
        )

        full_prompt = f"""{RAG_SYSTEM_PROMPT}

Conversation so far:
{history_block}

Context:
{context_block}

Question: {message}

Answer:"""

        try:
            for chunk in self.client.models.generate_content_stream(
                model=CHAT_MODEL, contents=full_prompt
            ):
                if chunk.text:
                    yield chunk.text
        except genai_errors.ClientError as e:
            raise RuntimeError(self._friendly_error(e)) from e

    @staticmethod
    def _friendly_error(err: genai_errors.ClientError) -> str:
        msg = str(err)
        code = getattr(err, "code", 0) or 0
        if not code:
            code_match = re.search(r"^(\d+)", msg)
            code = int(code_match.group(1)) if code_match else 0
        if code == 429:
            retry_match = re.search(r"retry\s+in\s+([\d.]+)s", msg, re.I)
            wait = retry_match.group(1) if retry_match else None
            base = "The AI service is temporarily out of requests due to a quota limit."
            if wait:
                return f"{base} Please wait about {float(wait):.0f} seconds before trying again."
            return f"{base} Please wait a moment and try again."
        if code == 403:
            return "The AI service couldn't authenticate your API key. Please check your configuration."
        if code == 400:
            return "The request to the AI service was invalid. Try rephrasing your question."
        if code == 404:
            return "The AI model is not available. The service may be updating."
        return "The AI service returned an unexpected error. Please try again later."

    def generate_title(self, message: str) -> str:
        prompt = f"Summarize this question in 3-6 words as a concise chat title. Just the title, no quotes or punctuation:\n\n{message}"
        try:
            resp = self.client.models.generate_content(model=CHAT_MODEL, contents=prompt)
            title = resp.text.strip().strip('"').strip("'")
            return title[:60]
        except Exception:
            return message[:40] + ("..." if len(message) > 40 else "")
