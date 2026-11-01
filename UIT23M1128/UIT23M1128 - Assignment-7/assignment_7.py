# Name : Divyanshu Girijesh Singh
# Class : B.Tech IT B - B3
# PRN : UIT23M1128
#  Assignment 7: Sentiment Analysis using Machine Learningss

"""
Assignment 7: Sentiment Analysis using Machine Learnings
- Binary sentiment classifier (positive/negative) on customer product reviews
- Scikit-learn pipelines: TF-IDF + Logistic Regression, Naive Bayes, SVM
- Model evaluation: Accuracy, Confusion Matrix, ROC-AUC curves
- FMCG Brand-Sentiment monitoring system triggering real-time alerts when negative spikes > 30%
"""

import os
import csv
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def train_and_evaluate_classifiers(texts: List[str], labels: List[int]) -> Dict[str, dict]:
    """Trains Logistic Regression, Naive Bayes, and SVM on TF-IDF vectors and computes metrics."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine (SVM)": SVC(probability=True)
    }

    results = {}
    for name, model in models.items():
        model.fit(X, labels)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds

        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)
        roc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 1.0

        results[name] = {
            "model": model,
            "accuracy": acc,
            "confusion_matrix": cm,
            "roc_auc": roc,
            "predictions": preds
        }
    return results

def monitor_brand_health(predictions: List[int], threshold_percent: float = 30.0) -> str:
    """Evaluates percentage of negative reviews and triggers brand risk alert if threshold exceeded."""
    total = len(predictions)
    if total == 0:
        return "NO DATA"
    neg_count = sum(1 for p in predictions if p == 0)
    neg_ratio = (neg_count / total) * 100

    if neg_ratio > threshold_percent:
        return f"CRITICAL ALERT: Negative sentiment spike at {neg_ratio:.1f}% (> {threshold_percent}%)"
    return f"NORMAL: Negative sentiment ratio at {neg_ratio:.1f}%"

def main():
    print("=" * 60)
    print("ASSIGNMENT 7: SENTIMENT ANALYSIS USING MACHINE LEARNING")
    print("=" * 60)

    csv_path = os.path.join(os.path.dirname(__file__), "fmcg_product_reviews.csv")
    if not os.path.exists(csv_path):
        print(f"Data file not found at {csv_path}")
        return

    reviews, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviews.append(row["review"])
            labels.append(int(row["sentiment"]))

    print(f"\nLoaded {len(reviews)} product review samples.")

    # 1. Train and Evaluate Classifiers
    eval_results = train_and_evaluate_classifiers(reviews, labels)
    print("\n--- Model Evaluation Benchmarks ---")
    print(f"{'Model Architecture':<30} | {'Accuracy':<10} | {'ROC-AUC':<10}")
    print("-" * 58)
    for name, res in eval_results.items():
        print(f"{name:<30} | {res['accuracy']*100:.1f}%      | {res['roc_auc']:.4f}")

    # 2. Real-Time Brand Health Alert Monitoring
    lr_preds = eval_results["Logistic Regression"]["predictions"]
    status_alert = monitor_brand_health(lr_preds, threshold_percent=30.0)
    print("\n--- Real-Time FMCG Brand Monitoring Dashboard ---")
    print(f"Status: {status_alert}")

if __name__ == "__main__":
    main()
