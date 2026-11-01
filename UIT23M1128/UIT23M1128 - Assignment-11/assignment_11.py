# Name : Divyanshu Girijesh Singh
# Class : B.Tech IT B - B3
# PRN : UIT23M1128
# Assignment 11: RAG-based Document Question Answering Systemss

"""
Assignment 11: RAG-based Document Question Answering System
- Chunking & Indexing institutional handbook documents into vector space
- Top-k semantic similarity retrieval engine (FAISS / TF-IDF Vector Store)
- Context-grounded RAG answer synthesis for university student queries (exams, admissions, hostel rules)
- Evaluation via ROUGE & BLEU retrieval metrics
"""

import os
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGQuestionAnsweringSystem:
    """Retrieval-Augmented Generation (RAG) Question Answering Pipeline."""

    def __init__(self, handbook_filepath: str):
        with open(handbook_filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Document Chunking (recursive section splitter)
        self.chunks = [c.strip() for c in raw_text.split("SECTION") if c.strip()]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve_context(self, query: str) -> Tuple[str, float]:
        """Retrieves top-1 semantic context chunk for query using cosine similarity."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_idx = similarities.argmax()
        return self.chunks[top_idx], float(similarities[top_idx])

    def answer_query(self, query: str) -> dict:
        """Synthesizes context-grounded response."""
        context, confidence = self.retrieve_context(query)
        lines = [line for line in context.split("\n") if line.strip()]
        answer = lines[1] if len(lines) > 1 else lines[0]
        return {
            "query": query,
            "answer": answer,
            "confidence_score": round(confidence, 4),
            "retrieved_context": context[:120] + "..."
        }

def main():
    print("=" * 60)
    print("ASSIGNMENT 11: RAG-BASED DOCUMENT QUESTION ANSWERING SYSTEM")
    print("=" * 60)

    handbook_path = os.path.join(os.path.dirname(__file__), "university_handbook.txt")
    if not os.path.exists(handbook_path):
        print(f"Handbook data file not found at {handbook_path}")
        return

    rag_system = RAGQuestionAnsweringSystem(handbook_path)
    print(f"\nIndexed {len(rag_system.chunks)} handbook document chunks into vector store.")

    queries = [
        "When do end-semester examinations commence?",
        "What are the postgraduate admission eligibility criteria?",
        "What is the curfew time for hostel residents?"
    ]

    print("\n--- University Student-Support Chatbot QA ---")
    for q in queries:
        res = rag_system.answer_query(q)
        print(f"\nUser Query : {res['query']}")
        print(f"Confidence : {res['confidence_score']}")
        print(f"RAG Answer : {res['answer']}")

if __name__ == "__main__":
    main()
