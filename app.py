"""
AI Customer Support Ticket Triage - Flask Web Application
=========================================================
This module serves the web interface and handles inference requests.
Features:
- Responsive, modern UI built with Bootstrap 5 and custom styles.
- One-click sample tickets to demonstrate various categories and priorities.
- Real-time classification showing Category, Priority, Queue, Confidence, and Status.
- Self-contained using Flask's render_template_string to keep the project
  structure clean and portable without requiring subdirectories.
"""

import os
from flask import Flask, request, render_template_string, jsonify
from model import predict_ticket, load_model

app = Flask(__name__)

# Preload or train model on server startup to ensure instant first response
try:
    load_model()
except Exception as e:
    print(f"[!] Warning during initial model loading: {e}")

# HTML Template embedded directly for a clean, self-contained single-file web app
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Customer Support Ticket Triage</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Google Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            --card-bg: rgba(255, 255, 255, 0.98);
        }

        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-gradient);
            min-height: 100vh;
            color: #1e293b;
            padding-bottom: 3rem;
        }

        .header-section {
            padding: 3.5rem 1rem 2rem 1rem;
            text-align: center;
            color: #ffffff;
        }

        .badge-pill-header {
            background: rgba(99, 102, 241, 0.25);
            border: 1px solid rgba(165, 180, 252, 0.4);
            color: #c7d2fe;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.4rem 1rem;
            border-radius: 50rem;
            display: inline-block;
            margin-bottom: 1rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .main-card {
            background: var(--card-bg);
            border-radius: 1.25rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow: hidden;
        }

        .form-control:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 0.25rem rgba(79, 70, 229, 0.2);
        }

        .btn-predict {
            background-color: var(--primary);
            border: none;
            color: white;
            font-weight: 600;
            padding: 0.85rem 1.75rem;
            border-radius: 0.75rem;
            transition: all 0.2s ease;
        }

        .btn-predict:hover {
            background-color: var(--primary-hover);
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4);
        }

        .sample-btn {
            font-size: 0.825rem;
            border-radius: 0.5rem;
            background-color: #f1f5f9;
            color: #475569;
            border: 1px solid #cbd5e1;
            padding: 0.35rem 0.75rem;
            margin: 0.25rem;
            transition: all 0.15s ease;
            text-align: left;
        }

        .sample-btn:hover {
            background-color: #e2e8f0;
            color: #0f172a;
            border-color: #94a3b8;
        }

        .result-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 1rem;
            padding: 1.75rem;
            animation: fadeIn 0.3s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .metric-card {
            background: #ffffff;
            border-radius: 0.75rem;
            padding: 1.1rem;
            border: 1px solid #e2e8f0;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .metric-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
        }

        /* Priority Badge Colors */
        .priority-high {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
        .priority-medium {
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #fde68a;
        }
        .priority-low {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
        }

        /* Status Badges */
        .status-auto {
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        .status-review {
            background-color: #ffedd5;
            color: #9a3412;
            border: 1px solid #fed7aa;
        }

        .badge-custom {
            padding: 0.35rem 0.75rem;
            border-radius: 0.5rem;
            font-weight: 700;
            font-size: 0.95rem;
            display: inline-block;
        }

        .progress {
            height: 0.65rem;
            border-radius: 1rem;
            background-color: #e2e8f0;
        }
    </style>
</head>
<body>
    <div class="container py-4">
        <!-- Header -->
        <div class="header-section">
            <span class="badge-pill-header">Machine Learning Powered • NLP System</span>
            <h1 class="display-5 fw-bold mb-3">AI Customer Support Ticket Triage</h1>
            <p class="lead text-light opacity-75 mx-auto" style="max-width: 650px;">
                Automatically categorize incoming customer inquiries, assess operational urgency,
                assign to the right support queue, and flag tickets requiring human review.
            </p>
        </div>

        <!-- Main Card -->
        <div class="row justify-content-center">
            <div class="col-lg-9">
                <div class="main-card p-4 p-md-5">
                    <!-- Form -->
                    <form method="POST" action="/" id="triageForm">
                        <div class="mb-4">
                            <label for="ticket_text" class="form-label fw-bold text-dark fs-5">
                                Enter Customer Support Ticket
                            </label>
                            <textarea 
                                class="form-control" 
                                id="ticket_text" 
                                name="ticket_text" 
                                rows="4" 
                                placeholder="Describe the customer issue here (e.g., 'I was charged twice on my credit card this month...')" 
                                required>{{ ticket_text or '' }}</textarea>
                        </div>

                        <!-- Sample Ticket Quick Buttons -->
                        <div class="mb-4">
                            <div class="small fw-bold text-muted mb-2">⚡ Quick Test Samples (Click to populate):</div>
                            <div class="d-flex flex-wrap">
                                <button type="button" class="btn sample-btn" onclick="fillTicket('I was charged twice for my subscription this month. Please issue a refund immediately.')">
                                    💳 Double Charge (Billing)
                                </button>
                                <button type="button" class="btn sample-btn" onclick="fillTicket('The entire web application is down and returning HTTP 500 internal server error.')">
                                    🚨 Server Outage (Technical)
                                </button>
                                <button type="button" class="btn sample-btn" onclick="fillTicket('My account has been locked due to too many failed login attempts. Please unlock it.')">
                                    🔒 Account Locked (Account)
                                </button>
                                <button type="button" class="btn sample-btn" onclick="fillTicket('Would love to see a native integration with Slack and Microsoft Teams notifications.')">
                                    💡 Feature Request (Product)
                                </button>
                                <button type="button" class="btn sample-btn" onclick="fillTicket('Hello, I was wondering about something.')">
                                    ❓ Ambiguous Ticket (Needs Review)
                                </button>
                            </div>
                        </div>

                        <!-- Action Button -->
                        <div class="d-grid gap-2 d-md-flex justify-content-md-between align-items-center mb-4">
                            <button type="submit" class="btn btn-predict">
                                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" class="bi bi-cpu me-1" viewBox="0 0 16 16">
                                  <path d="M5 0a.5.5 0 0 1 .5.5V2h1V.5a.5.5 0 0 1 1 0V2h1V.5a.5.5 0 0 1 1 0V2h1V.5a.5.5 0 0 1 1 0V2A2.5 2.5 0 0 1 14 4.5h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14v1h1.5a.5.5 0 0 1 0 1H14a2.5 2.5 0 0 1-2.5 2.5v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14h-1v1.5a.5.5 0 0 1-1 0V14A2.5 2.5 0 0 1 2 11.5H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2v-1H.5a.5.5 0 0 1 0-1H2A2.5 2.5 0 0 1 4.5 2V.5A.5.5 0 0 1 5 0m-.5 3A1.5 1.5 0 0 0 3 4.5v7A1.5 1.5 0 0 0 4.5 13h7a1.5 1.5 0 0 0 1.5-1.5v-7A1.5 1.5 0 0 0 11.5 3zM5 6.5A1.5 1.5 0 0 1 6.5 5h3A1.5 1.5 0 0 1 11 6.5v3A1.5 1.5 0 0 1 9.5 11h-3A1.5 1.5 0 0 1 5 9.5z"/>
                                </svg>
                                Predict Ticket Triage
                            </button>
                            <span class="text-muted small">TF-IDF + Logistic Regression Classifier</span>
                        </div>
                    </form>

                    <!-- Prediction Results Section -->
                    {% if result %}
                    <hr class="my-4 text-muted">
                    <div class="result-box">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5 class="fw-bold m-0 text-dark">Triage Classification Results</h5>
                            {% if result.status == 'Auto Classified' %}
                                <span class="badge-custom status-auto">✓ {{ result.status }}</span>
                            {% else %}
                                <span class="badge-custom status-review">⚠️ {{ result.status }}</span>
                            {% endif %}
                        </div>

                        <div class="row g-3">
                            <!-- Category -->
                            <div class="col-sm-6 col-md-3">
                                <div class="metric-card">
                                    <div class="metric-label">Predicted Category</div>
                                    <div class="metric-value text-primary">{{ result.category }}</div>
                                </div>
                            </div>

                            <!-- Priority -->
                            <div class="col-sm-6 col-md-3">
                                <div class="metric-card">
                                    <div class="metric-label">Priority Level</div>
                                    <div>
                                        {% if result.priority == 'High' %}
                                            <span class="badge-custom priority-high">High</span>
                                        {% elif result.priority == 'Medium' %}
                                            <span class="badge-custom priority-medium">Medium</span>
                                        {% else %}
                                            <span class="badge-custom priority-low">Low</span>
                                        {% endif %}
                                    </div>
                                </div>
                            </div>

                            <!-- Queue -->
                            <div class="col-sm-6 col-md-3">
                                <div class="metric-card">
                                    <div class="metric-label">Assigned Queue</div>
                                    <div class="metric-value text-dark" style="font-size: 1.05rem;">{{ result.queue }}</div>
                                </div>
                            </div>

                            <!-- Confidence -->
                            <div class="col-sm-6 col-md-3">
                                <div class="metric-card">
                                    <div class="metric-label">Confidence Score</div>
                                    <div class="metric-value text-dark mb-1">{{ result.confidence_pct }}</div>
                                    <div class="progress">
                                        <div class="progress-bar {% if result.confidence >= 0.60 %}bg-success{% else %}bg-warning{% endif %}" 
                                             role="progressbar" 
                                             style="width: {{ result.confidence * 100 }}%;" 
                                             aria-valuenow="{{ result.confidence * 100 }}" 
                                             aria-valuemin="0" 
                                             aria-valuemax="100">
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Decision Explanation -->
                        <div class="mt-3 p-3 bg-white rounded border small text-muted">
                            <strong>Triage Decision Summary:</strong>
                            Ticket routed to <strong>{{ result.queue }}</strong> with 
                            <strong>{{ result.priority }}</strong> urgency. 
                            {% if result.status == 'Auto Classified' %}
                                The prediction confidence ({{ result.confidence_pct }}) satisfies the automated threshold (&ge; 60%).
                            {% else %}
                                The prediction confidence ({{ result.confidence_pct }}) is below the 60% threshold, flagging this ticket for human specialist evaluation.
                            {% endif %}
                        </div>
                    </div>
                    {% endif %}
                </div>

                <!-- Footer details -->
                <div class="text-center text-white-50 mt-4 small">
                    AI Customer Support Ticket Triage • Powered by Scikit-Learn & Flask • Project 1
                </div>
            </div>
        </div>
    </div>

    <!-- Interactive script for sample buttons -->
    <script>
        function fillTicket(text) {
            document.getElementById('ticket_text').value = text;
        }
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Renders main page and handles form submission for ticket prediction.
    """
    result = None
    ticket_text = ""

    if request.method == "POST":
        ticket_text = request.form.get("ticket_text", "").strip()
        if ticket_text:
            result = predict_ticket(ticket_text)

    return render_template_string(HTML_TEMPLATE, result=result, ticket_text=ticket_text)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic inference.
    Accepts JSON: {"ticket": "text content"}
    Returns JSON: {"category": ..., "priority": ..., "queue": ..., "confidence": ..., "status": ...}
    """
    data = request.get_json(silent=True) or {}
    ticket_text = data.get("ticket", "")
    if not ticket_text:
        return jsonify({"error": "Missing 'ticket' field in request JSON"}), 400

    prediction = predict_ticket(ticket_text)
    return jsonify(prediction), 200


if __name__ == "__main__":
    # Run locally on port 5000
    print("[*] Starting AI Customer Support Ticket Triage server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
