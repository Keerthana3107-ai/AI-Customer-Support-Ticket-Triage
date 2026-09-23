"""
AI Customer Support Ticket Triage - Machine Learning Pipeline
=============================================================
This module handles:
1. Loading and cleaning customer support ticket data.
2. Converting raw text into numerical features using TF-IDF Vectorization.
3. Training classification models for ticket Category and Priority.
4. Evaluating model performance with accuracy and classification reports.
5. Saving and loading trained model artifacts using Joblib.
6. Real-time inference with confidence thresholding for human review.
"""

import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths are configured relative to this script for cross-platform portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ticket_triage_model.joblib")

# Category to Support Queue Mapping
QUEUE_MAPPING = {
    "Billing": "Billing Support",
    "Technical": "Technical Support",
    "Account": "Account Support",
    "Product": "Product Support"
}

# Confidence threshold below which a ticket requires manual human inspection
CONFIDENCE_THRESHOLD = 0.60


def clean_text(text: str) -> str:
    """
    Cleans raw ticket text by lowercasing, stripping special characters,
    and removing redundant whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove special characters while preserving standard alphanumeric words
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_preprocess_data():
    """
    Loads dataset.csv and applies text cleaning.
    Returns cleaned pandas DataFrame.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    df["clean_ticket"] = df["ticket"].apply(clean_text)
    return df


def train_and_evaluate_models():
    """
    Trains the TF-IDF vectorizer and Logistic Regression classifiers for both
    Category and Priority, prints evaluation metrics, and persists the models.
    """
    print("=" * 60)
    print("AI Customer Support Ticket Triage - Model Training")
    print("=" * 60)

    # 1. Load data
    df = load_and_preprocess_data()
    print(f"[+] Loaded {len(df)} sample tickets from dataset.csv")

    X = df["clean_ticket"]
    y_category = df["category"]
    y_priority = df["priority"]

    # 2. Train/Test split for honest evaluation (80% train, 20% test)
    X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test = train_test_split(
        X, y_category, y_priority, test_size=0.20, random_state=42, stratify=y_category
    )

    # 3. TF-IDF Feature Extraction
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=1500
    )

    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)

    # 4. Train Category Classifier with C=10.0 for balanced probability calibration
    print("\n--- Training Category Classifier ---")
    cat_model = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    cat_model.fit(X_train_vec, y_cat_train)
    cat_pred = cat_model.predict(X_test_vec)
    cat_acc = accuracy_score(y_cat_test, cat_pred)
    print(f"Category Validation Accuracy: {cat_acc * 100:.2f}%\n")
    print("Category Classification Report:")
    print(classification_report(y_cat_test, cat_pred, zero_division=0))

    # 5. Train Priority Classifier
    print("--- Training Priority Classifier ---")
    prio_model = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    prio_model.fit(X_train_vec, y_prio_train)
    prio_pred = prio_model.predict(X_test_vec)
    prio_acc = accuracy_score(y_prio_test, prio_pred)
    print(f"Priority Validation Accuracy: {prio_acc * 100:.2f}%\n")
    print("Priority Classification Report:")
    print(classification_report(y_prio_test, prio_pred, zero_division=0))

    # 6. Fit final models on full dataset for optimal operational performance
    print("--- Retraining on Full Dataset for Best Production Quality ---")
    full_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=1500
    )
    X_full_vec = full_vectorizer.fit_transform(X)

    final_cat_model = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    final_cat_model.fit(X_full_vec, y_category)

    final_prio_model = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    final_prio_model.fit(X_full_vec, y_priority)

    # 7. Save model bundle using Joblib
    model_bundle = {
        "vectorizer": full_vectorizer,
        "category_model": final_cat_model,
        "priority_model": final_prio_model
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"[OK] Saved model bundle to: {MODEL_PATH}")
    print("=" * 60)

    return model_bundle


def load_model():
    """
    Loads the saved model bundle. If it doesn't exist, trains it automatically.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model file '{MODEL_PATH}' not found. Training model now...")
        return train_and_evaluate_models()
    return joblib.load(MODEL_PATH)


def predict_ticket(ticket_text: str):
    """
    Accepts raw ticket text and returns triage prediction:
    - category
    - priority
    - queue
    - confidence (percentage & decimal)
    - status ('Auto Classified' vs 'Needs Human Review')
    """
    if not ticket_text or not ticket_text.strip():
        return {
            "category": "Unknown",
            "priority": "Low",
            "queue": "General Support",
            "confidence": 0.0,
            "status": "Needs Human Review",
            "confidence_pct": "0.0%"
        }

    # Load model bundle
    bundle = load_model()
    vectorizer = bundle["vectorizer"]
    cat_model = bundle["category_model"]
    prio_model = bundle["priority_model"]

    # Preprocess and vectorize
    cleaned = clean_text(ticket_text)
    vec = vectorizer.transform([cleaned])

    # Predict category and confidence probability
    cat_probs = cat_model.predict_proba(vec)[0]
    cat_classes = cat_model.classes_
    max_idx = cat_probs.argmax()
    predicted_category = cat_classes[max_idx]
    confidence_score = float(cat_probs[max_idx])

    # Predict priority
    predicted_priority = prio_model.predict(vec)[0]

    # Map to support queue
    queue = QUEUE_MAPPING.get(predicted_category, "General Support")

    # Apply confidence threshold rule
    if confidence_score < CONFIDENCE_THRESHOLD:
        status = "Needs Human Review"
    else:
        status = "Auto Classified"

    return {
        "category": predicted_category,
        "priority": predicted_priority,
        "queue": queue,
        "confidence": round(confidence_score, 4),
        "confidence_pct": f"{confidence_score * 100:.1f}%",
        "status": status
    }


if __name__ == "__main__":
    # Train and evaluate models when executed directly
    train_and_evaluate_models()

    # Quick demo verification tests
    test_samples = [
        "My internet connection is not working and I cannot access the application.",
        "I was charged twice on my credit card for this month subscription.",
        "Could you please add an export to CSV feature for analytics?",
        "Hello, I need help with something."  # Intentionally vague for low confidence
    ]

    print("\nRunning Verification Samples:")
    print("-" * 60)
    for sample in test_samples:
        result = predict_ticket(sample)
        print(f"Ticket: \"{sample}\"")
        print(f" -> Category  : {result['category']}")
        print(f" -> Priority  : {result['priority']}")
        print(f" -> Queue     : {result['queue']}")
        print(f" -> Confidence: {result['confidence_pct']}")
        print(f" -> Status    : {result['status']}\n")
