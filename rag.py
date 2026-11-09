import glob
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICY_DIR = "data/policies"
CHUNK_SIZE = 40


# ---------- Task 1: chunking ----------
def chunk_documents(chunk_size=CHUNK_SIZE, overlap=10):
    step = chunk_size - overlap
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        name = os.path.basename(path)
        start = 0
        while True:
            end = start + chunk_size
            chunks.append({"source": name, "text": " ".join(words[start:end])})
            if end >= len(words):          # this chunk reaches the last word: stop
                break
            start += step
    return chunks


# ---------- Task 2: TF-IDF index and retrieval ----------
class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def retrieve(self, query, k=3):
        q = self.vectorizer.transform([query])           # same fitted vectorizer
        scores = cosine_similarity(q, self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(scores[i]),
                 "text": self.chunks[i]["text"]} for i in top]


INDEX = Index(chunk_documents(CHUNK_SIZE, 10))


def retrieve(query, k=3):
    return INDEX.retrieve(query, k)


# ---------- Task 3: augment and answer ----------
POLICY_SYSTEM = ("You are a POLICY ASSISTANT for a quick-commerce company. Answer only from the "
                 "supplied context. If the context does not contain the answer, say that you could "
                 "not find the information.")


def answer(question, index=None):
    index = index or INDEX
    chunks = index.retrieve(question, k=3)
    context = "\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    response = mock_llm.chat([{"role": "system", "content": POLICY_SYSTEM},
                              {"role": "user", "content": user}])
    return response.choices[0].message.content, [c["source"] for c in chunks]


# ---------- Task 4: evaluation ----------
def evaluate(index, label):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    rows = []
    for _, r in ev.iterrows():
        text, sources = answer(r["question"], index)
        rows.append({"question": r["question"], "expected_doc": r["expected_doc"],
                     "expected_fact": r["expected_fact"], "text": text, "sources": sources})
    answerable = [r for r in rows if r["expected_doc"] != "NONE"]
    unanswerable = [r for r in rows if r["expected_doc"] == "NONE"]
    doc_hits = sum(r["expected_doc"] in r["sources"] for r in answerable)
    fact_hits = sum(r["expected_fact"] in r["text"] for r in answerable)
    fallback = sum(r["text"].startswith("I could not find") for r in unanswerable)
    print(f"\n=== {label} ===")
    print(f"answerable: expected_doc in top 3: {doc_hits}/{len(answerable)} | "
          f"answer contains expected_fact: {fact_hits}/{len(answerable)}")
    print(f"unanswerable: fallback answer: {fallback}/{len(unanswerable)}")
    for r in answerable:
        if r["expected_doc"] not in r["sources"] or r["expected_fact"] not in r["text"]:
            print("  FAILS:", r["question"], "| expected fact:", r["expected_fact"])
            print("        answer:", r["text"])
    return rows


if __name__ == "__main__":
    chunks = chunk_documents(40, 10)
    counts = pd.Series([c["source"] for c in chunks]).value_counts().sort_index()
    print("total chunks:", len(chunks))
    print(counts.to_string())

    top = retrieve("refund policy late delivery")
    print("\ntop 3 for 'refund policy late delivery':")
    for t in top:
        print(" ", t["source"], round(t["score"], 3))

    evaluate(INDEX, "WITH OVERLAP 10")
    evaluate(Index(chunk_documents(40, 0)), "WITH OVERLAP 0")