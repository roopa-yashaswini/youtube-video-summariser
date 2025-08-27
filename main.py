from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from typing import List, Sequence, Union, Tuple
import textwrap
from urllib.parse import urlparse, parse_qs

load_dotenv(override=True)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_video_title(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string if soup.title else "Untitled Video"
        return title.replace(" - YouTube", "").strip(), False
    except Exception:
        return "Couldnt fetch the title", True

def get_video_id(url):
    u = urlparse(url)
    if u.hostname == "youtu.be":
        return u.path.lstrip("/"), False
    video_id = parse_qs(u.query).get("v")
    if video_id:
        return video_id[0], False
    return ValueError("Invalid Youtube URL: No Video ID found."), True

def get_youtube_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.get_transcript(video_id)
        # Join all text segments
        full_text = ' '.join([entry['text'] for entry in transcript])

        return full_text, False
    except (TranscriptsDisabled, NoTranscriptFound):
        return "Transcript not available for this video.", True

def chunk_documents(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    if chunk_size <= 0 or chunk_overlap <= 0:
        raise ValueError("Size and Overlap should greater than 0.")
    
    if chunk_overlap > chunk_size:
        raise ValueError("Overlap should be less than Chunk size.")
    
    if len(text) == 0:
        return []
    
    words = text.split()
    chunks = []

    step = chunk_size - chunk_overlap

    for i in range(0, len(words), step):
        chunk = words[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if i+chunk_size >= len(words):
            break
    return chunks

def embed_chunks(chunks: List[str], model: str) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    
    resp = openai.embeddings.create(model=model, input=chunks)
    embeddings = np.array([e.embedding for e in resp.data], dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings/norms

    return embeddings

class Retriever:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors
        self.n, self.d = vectors.shape

        self._index = faiss.IndexFlatIP(self.d)
        self._index.add(vectors)

    def _normalise_query(self, vec: np.ndarray) -> np.ndarray:
        v = vec.astype(np.float32, copy = False)
        n = np.linalg.norm(v) + 1e-12
        return v/n
    
    def search(self, query_vec: np.ndarray, top_k: int = 5)-> List[Tuple[int, float]]:
        if self.n == 0:
            return []
        
        q = query_vec.reshape(-1).astype(np.float32)
        q = self._normalise_query(q)

        k = min(top_k, self.n)

        scores, indices = self._index.search(q[None, :], k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0])]
    
def build_and_retrieve_from_vectorDB(vectors: np.ndarray) -> Retriever :
    return Retriever(vectors)

def extract_relevant_vectors(question: str, retriever: Retriever, chunks: List[str], top_k: int) -> List[str]:
    if len(question) == 0 or retriever is None:
        return []

    q_vec = embed_chunks([question], "text-embedding-3-small")
    top_k_res = retriever.search(q_vec[0], top_k)
    top_k_chunks = [chunks[i] for i, _ in top_k_res]

    return top_k_chunks


    
class VideoChatbot:
    def __init__(self, url):
        self.url = url
        self.title = get_video_title(url)
        self.video_id = get_video_id(url)
        self.transcript = get_youtube_transcript(self.video_id)
        self.chunks = chunk_documents(self.transcript)
        self.embeddings = embed_chunks(self.chunks, "text-embedding-3-small")
        self.retriever = build_and_retrieve_from_vectorDB(self.embeddings)
        self.history = []
        self.system_prompt = {
            "role": "system",
            "content": f"You are an expert assistant that analyzes the transcript of youtube video and who only answers questions using the transcript of the YouTube video '{self.title}'. When relevant, answer with specific details from the transcript."
        }
    
    def summarize(self) -> str:

        prompt = f"""
        Summarize the following YouTube video transcript in detail:\n\n{self.transcript}\n\n Provide:  
        1) **TL;DR** bullets (3–6 lines)
        2) **Key sections** (use headings; include timestamps only if present)
        3) **Actionable takeaways**
        4) **Notable quotes or definitions**
        Answer in Markdown."""
        messages = [self.system_prompt, {"role": "user", "content": prompt}]
        resp = openai.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        summary = resp.choices[0].message.content
        self.history = [self.system_prompt, {"role": "user", "content": prompt}, {"role": "assistant", "content": summary}]
        return summary

    def chat(self, question: str, max_chrs: int = 8000) -> str:
        if not question.strip():
            return "Please ask a question."
        
        retrieved_chunks = extract_relevant_vectors(question, self.retriever, self.chunks, top_k=5)
        if not retrieved_chunks:
            return "I couldn't find relevant context in the transcript for that question."
        retrieved_cntxt = "\n\n".join(retrieved_chunks)
        retrieved_cntxt = retrieved_cntxt[:max_chrs]

        prompt = textwrap.dedent(f"""
            Context from the transcript (may be partial):
            {retrieved_cntxt}

            Question: {question}

            Instructions:
            - Cite or quote small snippets from the context when helpful.
            - If the answer is not present in the context, explicitly say so.
            - Keep the answer concise and in markdown.
        """).strip()

        # Keep just the last few turns to reduce token use
        trimmed_history = list(self.history)[-8:] if self.history else []

        messages = [self.system_prompt] + trimmed_history + [
            {"role": "user", "content": prompt}
        ]

        
        resp = openai.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        answer = resp.choices[0].message.content.strip()
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return answer
        


if __name__ == "__main__":
    url = input("Paste your YouTube URL: ")
    bot = VideoChatbot(url)
    print("----- SUMMARY -----")
    print(bot.summarize())
    print("\nNow, ask questions about the video!")
    while True:
        q = input("\nQ: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\nA:", bot.chat(q))