# Name : Divyanshu Girijesh Singh
# Class : B.Tech IT B - B3
# PRN : UIT23M1128
# Assignment 5: Topic Modeling using LSA / LDA 




"""
Assignment 5: Topic Modeling using LSA / LDA
- Topic Modeling using Latent Semantic Analysis (LSA / SVD) and Latent Dirichlet Allocation (LDA)
- Coherence & Topic Keyword Extraction
- Real-World Application: Analyzing Telecom Support Tickets for operational decision-making
"""

import os
import csv
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

def fit_lda_topics(corpus: List[str], num_topics: int = 3, top_words_count: int = 5) -> List[Tuple[str, List[str]]]:
    """Fits Latent Dirichlet Allocation (LDA) topic model and returns top keywords per topic."""
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    topics = []
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-top_words_count - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append((f"Topic {idx + 1}", top_words))
    return topics

def fit_lsa_topics(corpus: List[str], num_topics: int = 3, top_words_count: int = 5) -> List[Tuple[str, List[str]]]:
    """Fits Latent Semantic Analysis (LSA / TruncatedSVD) topic model."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    lsa = TruncatedSVD(n_components=num_topics, random_state=42)
    lsa.fit(tfidf_matrix)

    topics = []
    for idx, comp in enumerate(lsa.components_):
        top_indices = comp.argsort()[:-top_words_count - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append((f"LSA Topic {idx + 1}", top_words))
    return topics

def main():
    print("=" * 60)
    print("ASSIGNMENT 5: TOPIC MODELING USING LSA / LDA")
    print("=" * 60)

    csv_path = os.path.join(os.path.dirname(__file__), "telecom_support_tickets.csv")
    if not os.path.exists(csv_path):
        print(f"Data file not found at {csv_path}")
        return

    documents = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            documents.append(row["text"])

    print(f"\nLoaded {len(documents)} customer support ticket descriptions.")

    # 1. Fit LDA Model
    lda_results = fit_lda_topics(documents, num_topics=3)
    print("\n--- Discovered Topics via Latent Dirichlet Allocation (LDA) ---")
    for topic_name, words in lda_results:
        print(f"{topic_name:<12}: {', '.join(words)}")

    # 2. Fit LSA Model
    lsa_results = fit_lsa_topics(documents, num_topics=3)
    print("\n--- Discovered Topics via Latent Semantic Analysis (LSA) ---")
    for topic_name, words in lsa_results:
        print(f"{topic_name:<12}: {', '.join(words)}")

if __name__ == "__main__":
    main()
