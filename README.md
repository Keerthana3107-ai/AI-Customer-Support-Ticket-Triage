# AI Customer Support Ticket Triage

An end-to-end Machine Learning web application designed to automatically classify, prioritize, and route incoming customer support tickets, while routing low-confidence predictions to human agents.

Built with **Python**, **Scikit-learn**, and **Flask**, this project demonstrates fundamental Natural Language Processing (NLP) techniques, multi-target classification, probability-based decision thresholds, and practical web integration in an accessible and clean codebase.

---

## 📌 Project Overview

Customer support centers handle thousands of unstructured support queries daily. Manually reading each ticket to assign categories and determine urgency causes response delays.

This project solves this operational challenge by reading raw ticket text and determining:
1. **Ticket Category**: What area is affected? (`Billing`, `Technical`, `Account`, `Product`)
2. **Ticket Priority**: How urgently should it be resolved? (`Low`, `Medium`, `High`)
3. **Support Queue**: Which team should handle the ticket? (`Billing Support`, `Technical Support`, `Account Support`, `Product Support`)
4. **Confidence Score**: How confident is the ML model in its classification?
5. **Review Status**: If confidence is below **60%**, the ticket is flagged as **"Needs Human Review"**, otherwise **"Auto Classified"**.

---

## ✨ Features

- **Multi-Output Triage**: Predicts both ticket category and priority urgency simultaneously.
- **Support Queue Mapping**: Automatically routes tickets to the appropriate department.
- **Confidence Scoring & Human-in-the-Loop Thresholding**: Prevents misrouting by flagging ambiguous inquiries with $< 60\%$ confidence.
- **Modern & Responsive UI**: Clean Bootstrap 5 interface featuring glassmorphic design, priority-colored badges, confidence progress bars, and one-click quick test samples.
- **REST API Endpoint**: Offers both an interactive web dashboard and a JSON API (`/api/predict`) for automation.
- **Robust Model Fallback**: Automatically trains the model upon starting if pre-trained model files are missing.
- **Zero External API Dependencies**: Operates completely offline with local Python packages.

---

## 🛠️ Technologies Used

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python (3.8+) | Core programming language |
| **Data Processing** | Pandas | Loading and structuring the ticket dataset |
| **Machine Learning** | Scikit-learn | TF-IDF feature extraction, Logistic Regression classifiers, and evaluation metrics |
| **Model Persistence** | Joblib | Serializing trained models and vectorizers to disk |
| **Web Server** | Flask | Lightweight WSGI web application framework and routing |
| **Frontend UI** | HTML5, CSS3, Bootstrap 5 | Responsive, beginner-friendly web interface |

---

## 📂 Project Structure

The project follows a minimal, self-contained file layout:

```
AI_Customer_Support_Ticket_Triage/
│
├── dataset.csv          # Sample dataset with 52 realistic support tickets
├── model.py            # Text preprocessing, TF-IDF, model training & evaluation
├── app.py              # Flask web server with embedded responsive UI
├── requirements.txt    # Python package dependencies
└── README.md           # Complete documentation and setup guide
```

---

## 🧠 How the Machine Learning Model Works

The ML pipeline inside `model.py` follows standard NLP and classification stages:

```
Raw Ticket Text ──> [Text Preprocessing] ──> [TF-IDF Vectorizer] ──> [Logistic Regression]
                                                                            │
                       ┌────────────────────────────────────────────────────┴──────────────────────────┐
                       ▼                                                                               ▼
              Category Classifier                                                             Priority Classifier
                       │                                                                               │
          Predicts Category & Confidence                                                          Predicts Priority
                       │                                                                               │
      ┌────────────────┴────────────────┐                                                              │
      ▼                                 ▼                                                              │
  Confidence ≥ 60%               Confidence < 60%                                                      │
         │                              │                                                              │
  "Auto Classified"           "Needs Human Review"                                                     │
         │                              │                                                              │
         └──────────────────────────────┴───────────────┬──────────────────────────────────────────────┘
                                                        ▼
                                       Final Triage Output (Category,
                                       Priority, Queue, Confidence)
```

1. **Text Preprocessing (`clean_text`)**:
   - Converts text to lowercase to ensure casing consistency (e.g., `"Billing"` == `"billing"`).
   - Removes special symbols while keeping letters and numbers.
   - Trims excess whitespace.

2. **Feature Extraction (TF-IDF Vectorizer)**:
   - **TF-IDF** (Term Frequency - Inverse Document Frequency) converts text into numerical vectors.
   - Measures how frequently a word appears in a ticket while penalizing common stopwords (like *"the"*, *"is"*, *"at"*).
   - Generates unigrams and bigrams (`ngram_range=(1, 2)`) to capture multi-word phrases like *"credit card"* or *"server error"*.

3. **Classification (Multinomial Logistic Regression)**:
   - Uses `LogisticRegression(max_iter=1000, random_state=42)` for its computational efficiency, interpretability, and probabilistic predictions.
   - Outputs calibrated probabilities through the Softmax function via `.predict_proba()`.

4. **Confidence Thresholding**:
   - The maximum probability across all categories represents the model's confidence.
   - If $\text{confidence} < 0.60$, the system conservatively flags the ticket for manual human review.

---

## 📊 Dataset Explanation

`dataset.csv` contains 52 realistic customer support tickets across all 4 categories and 3 priority levels:

- **Columns**:
  - `ticket`: Text description written by the customer.
  - `category`: `Billing`, `Technical`, `Account`, or `Product`.
  - `priority`: `Low`, `Medium`, or `High`.
  - `queue`: `Billing Support`, `Technical Support`, `Account Support`, or `Product Support`.

### Category Distribution
- **Billing**: Double charges, invoice requests, expired cards, subscription cancellation, refund requests.
- **Technical**: 500 internal server errors, gateway timeouts, app crashes, console errors, slow performance.
- **Account**: Password resets, locked accounts, 2FA setup, compromised logins, account deletion requests.
- **Product**: Feature requests, dark mode suggestions, integration inquiries, UI usability feedback.

---

## 🚀 Installation & Running Guide

Follow these step-by-step instructions to set up and run the project locally on your machine.

### Step 1: Open Terminal in Project Directory
Open PowerShell or Command Prompt and navigate to the project directory:

```bash
cd AI_Customer_Support_Ticket_Triage
```

### Step 2: Create a Virtual Environment

**On Windows (PowerShell / Command Prompt):**
```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

**On Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Train the Model

Run `model.py` to preprocess data, train the classifiers, display accuracy metrics, and save the model file (`ticket_triage_model.joblib`):

```bash
python model.py
```

*Sample console output:*
```
============================================================
AI Customer Support Ticket Triage - Model Training
============================================================
[+] Loaded 52 sample tickets from dataset.csv

--- Training Category Classifier ---
Category Validation Accuracy: 100.00%

--- Training Priority Classifier ---
Priority Validation Accuracy: 90.91%

[✓] Saved model bundle to: ticket_triage_model.joblib
============================================================
```

### Step 6: Start the Flask Web Application

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Example Test Tickets

### Example 1: Technical Issue
- **Input**:
  > *"The entire web application is down and returning HTTP 500 internal server error."*
- **Expected Output**:
  - **Category**: `Technical`
  - **Priority**: `High`
  - **Support Queue**: `Technical Support`
  - **Confidence**: `> 80%`
  - **Status**: `Auto Classified`

### Example 2: Billing Issue
- **Input**:
  > *"I was charged twice for my subscription this month. Please issue a refund immediately."*
- **Expected Output**:
  - **Category**: `Billing`
  - **Priority**: `High`
  - **Support Queue**: `Billing Support`
  - **Confidence**: `> 85%`
  - **Status**: `Auto Classified`

### Example 3: Ambiguous Query (Triggers Human Review)
- **Input**:
  > *"Hello, I was wondering about something."*
- **Expected Output**:
  - **Category**: (Diffused across classes)
  - **Confidence**: `< 60%`
  - **Status**: `Needs Human Review`

---

## 🔌 Programmatic REST API Usage

You can also send tickets programmatically via HTTP POST:

**cURL Request:**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d "{\"ticket\": \"My credit card was charged twice for the renewal.\"}"
```

**JSON Response:**
```json
{
  "category": "Billing",
  "priority": "High",
  "queue": "Billing Support",
  "confidence": 0.8841,
  "confidence_pct": "88.4%",
  "status": "Auto Classified"
}
```

---

## 🔮 Future Improvements

1. **Transformer / BERT Embeddings**: Use lightweight models like DistilBERT for improved semantic understanding of slang and domain jargon.
2. **Multi-Label Tagging**: Support tickets requiring multiple tags (e.g., simultaneous Billing and Technical bug).
3. **Sentiment Analysis**: Analyze customer sentiment to automatically escalate angry or frustrated tickets.
4. **Automated AI Draft Replies**: Integrate small generative models to propose draft email responses for human agents.
5. **Database Storage**: Store incoming tickets and agent resolution feedback in SQLite or PostgreSQL for continuous model retraining.

---

## 📄 License
This project is open-source under the MIT License and built for educational and demonstration purposes.
