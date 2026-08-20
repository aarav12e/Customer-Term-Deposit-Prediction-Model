# 🏦 Term Deposit Subscription Predictor

> A full-stack machine learning web app that predicts whether a bank customer will subscribe to a term deposit — built with a Random Forest classifier, FastAPI, and React.

---

## 📌 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [How the Model Works](#-how-the-model-works)
- [The Algorithm — Random Forest Explained](#-the-algorithm--random-forest-explained)
- [Features Used for Prediction](#-features-used-for-prediction)
- [Why Call Duration is Excluded](#-why-call-duration-is-excluded)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)

---

## 🎯 What This Project Does

This app predicts the likelihood of a bank customer subscribing to a **term deposit** (a fixed-interest savings product) based on their demographic profile and campaign history — *before* a phone call is even made.

A bank's marketing team can use this tool to:
- **Prioritise** which customers to call
- **Avoid** wasting time on very unlikely subscribers
- **Increase** campaign conversion rates

The model outputs:
- A **binary prediction** — "Likely to subscribe" or "Unlikely to subscribe"
- A **probability score** — e.g., 73.4% — so the team can rank customers by confidence

---

## 🧠 How the Model Works

The model is trained on the **UCI Bank Marketing Dataset**, which contains records of 45,211 phone-based marketing campaigns run by a Portuguese bank.

### Training Pipeline

The model is a **scikit-learn Pipeline** that chains two steps:

```
Raw Customer Data
       │
       ▼
┌─────────────────────────────────────┐
│  Step 1: ColumnTransformer          │
│                                     │
│  ┌──────────────────────────────┐   │
│  │ Numerical columns            │   │
│  │  → StandardScaler            │   │
│  │    (zero mean, unit variance)│   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │ Categorical columns          │   │
│  │  → OneHotEncoder             │   │
│  │    (converts text → numbers) │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Step 2: RandomForestClassifier     │
│  (the actual prediction model)      │
└─────────────────────────────────────┘
       │
       ▼
  Prediction + Probability
```

The entire pipeline — preprocessing and model — is saved as a single `.joblib` file. When the API loads it, it can accept raw customer data (with text fields like `"management"`, `"married"`, etc.) and produce a prediction in one call.

---

## 🌲 The Algorithm — Random Forest Explained

The core model is a **Random Forest Classifier**.

### What is a Decision Tree?

Before Random Forest, understand a **Decision Tree** — it works exactly like a flowchart:

```
                    Age > 50?
                   /          \
                YES             NO
               /                 \
       Balance > 3000?         Has Housing Loan?
        /        \               /            \
      YES          NO          YES              NO
       |            |           |               |
    LIKELY       UNLIKELY    UNLIKELY         LIKELY
```

Each internal node asks a yes/no question about a feature. The tree follows branches until it reaches a leaf — the prediction.

**Problem with a single tree:** It tends to memorise the training data (overfitting) and doesn't generalise well to new customers.

### How Random Forest Fixes This

Random Forest builds **hundreds of decision trees**, each trained slightly differently, and **combines their votes**:

```
Customer Data
      │
      ├──────► Tree  1 → "Likely"
      ├──────► Tree  2 → "Unlikely"
      ├──────► Tree  3 → "Likely"
      ├──────► Tree  4 → "Likely"
      │              ...
      └──────► Tree 100 → "Likely"

              Majority vote → "LIKELY"
              Probability   → 73%  (73 out of 100 trees said "Likely")
```

The two sources of randomness that make each tree different:
1. **Bootstrap sampling** — each tree is trained on a random sample of the data (with replacement)
2. **Feature subsampling** — at each split, the tree only considers a random subset of features, not all of them

This diversity means individual tree errors cancel each other out. The ensemble is far more robust than any single tree.

### Why Random Forest for This Problem?

| Property | Why it matters here |
|---|---|
| **Handles mixed data types** | Customer data has both numbers (age, balance) and categories (job, marital status) |
| **Robust to outliers** | Some customers have extreme balances; trees split by threshold, so outliers don't dominate |
| **Non-linear relationships** | The relationship between age and subscription is not linear — trees capture this naturally |
| **Feature importance** | We can see which features the model relies on most |
| **No strong assumptions** | Unlike Logistic Regression, Random Forest doesn't assume the data is linearly separable |

---

## 📊 Features Used for Prediction

The model uses **16 input features** (call duration is intentionally excluded — see next section):

### Customer Demographics

| Feature | Type | Example | Description |
|---|---|---|---|
| `age` | Integer | `42` | Customer's age in years |
| `job` | Category | `"management"` | Type of employment |
| `marital` | Category | `"married"` | Marital status |
| `education` | Category | `"tertiary"` | Highest education level |

### Financial Profile

| Feature | Type | Example | Description |
|---|---|---|---|
| `default` | yes/no | `"no"` | Has credit in default? |
| `balance` | Integer | `2341` | Average yearly account balance (€) |
| `housing` | yes/no | `"yes"` | Has a housing loan? |
| `loan` | yes/no | `"no"` | Has a personal loan? |

### Current Campaign

| Feature | Type | Example | Description |
|---|---|---|---|
| `contact` | Category | `"cellular"` | Communication channel |
| `day` | Integer | `15` | Last contact day of month |
| `month` | Category | `"oct"` | Last contact month |
| `campaign` | Integer | `2` | Number of contacts in this campaign |

### Previous Campaign History

| Feature | Type | Example | Description |
|---|---|---|---|
| `pdays` | Integer | `10` | Days since last contact in previous campaign (-1 = never) |
| `previous` | Integer | `3` | Number of contacts before this campaign |
| `poutcome` | Category | `"success"` | Outcome of the previous campaign |
| `was_contacted_before` | Integer | `1` | Engineered feature: 1 if contacted before, 0 if not |

### Key Signals the Model Relies On

Based on analysis of the dataset, the most predictive features are typically:
- **`poutcome = "success"`** — a previous successful campaign is the strongest positive signal
- **`month`** — October, March, September, December historically have higher conversion
- **`balance`** — higher savings balance = more likely to invest
- **`age`** — retirees (55+) subscribe at much higher rates
- **`campaign`** — fewer contacts = better (>5 calls dramatically drops probability)

---

## 🚫 Why Call Duration is Excluded

This is a critical design decision.

The UCI dataset includes a `duration` feature — the length of the last phone call in seconds. While it is **highly correlated** with the target (longer calls → much more likely to subscribe), it is **completely useless for pre-call prediction** because:

> You only know the call duration *after* the call has ended. By that point, you already know the outcome.

Including `duration` would make the model look far better in evaluation (a trivial model could get >90% accuracy), but it would be **data leakage** — the model would learn from information that doesn't exist at the time of decision-making.

This model is designed for a **real-world use case**: deciding *before* calling whether a customer is worth targeting. So `duration` is excluded entirely.

---

## 🏗 System Architecture

```
┌────────────────────────────────────────────────────────┐
│                     User's Browser                     │
│                                                        │
│   ┌──────────────────────────────────────────────┐     │
│   │         React + Vite Frontend                │     │
│   │  (form inputs → JSON payload → fetch POST)   │     │
│   └──────────────────────┬───────────────────────┘     │
└──────────────────────────┼─────────────────────────────┘
                           │  HTTP POST /predict
                           │  { age, job, marital, ... }
                           ▼
┌──────────────────────────────────────────────────────┐
│               FastAPI Backend (Python)               │
│                                                      │
│  1. Pydantic validates the incoming JSON             │
│  2. Converts fields to a pandas DataFrame (1 row)    │
│  3. Passes it through the joblib Pipeline:           │
│     a. ColumnTransformer preprocesses the row        │
│     b. RandomForestClassifier.predict_proba()        │
│  4. Returns { prediction, probability_percent, label}│
└──────────────────┬───────────────────────────────────┘
                   │
                   │  loads on startup
                   ▼
        term_deposit_model.joblib
        (sklearn Pipeline saved with joblib)
```

### Data Flow for a Single Prediction

```
User fills form
    │
    ▼
Frontend builds JSON:
{
  "age": 58,
  "job": "retired",
  "balance": 8500,
  "poutcome": "success",
  ...
}
    │
    ▼
FastAPI receives → Pydantic validates types & ranges
    │
    ▼
Convert to pandas DataFrame (single row)
    │
    ▼
Pipeline.predict_proba(row)
    ├── StandardScaler normalises: age=58 → 1.24, balance=8500 → 2.17
    ├── OneHotEncoder encodes:     job="retired" → [0,0,0,0,0,0,0,0,1,0,0,0]
    └── RandomForest: 100 trees vote → P(subscribe) = 0.82
    │
    ▼
Response:
{
  "prediction": "Likely to subscribe",
  "probability_percent": 82.0,
  "label": "yes"
}
    │
    ▼
Frontend renders probability bar + verdict badge
```

---

## 📁 Project Structure

```
project 2/
├── README.md                        ← You are here
│
├── backend/
│   ├── main.py                      ← FastAPI app (endpoints, schema, inference)
│   ├── requirements.txt             ← Python dependencies
│   ├── term_deposit_model.joblib    ← Trained sklearn Pipeline (not in git)
│   └── venv/                        ← Python virtual environment (not in git)
│
└── frontend/
    ├── index.html                   ← HTML entry point + font imports
    ├── vite.config.js               ← Vite bundler config
    ├── package.json                 ← Node dependencies
    └── src/
        ├── main.jsx                 ← React entry point
        ├── App.jsx                  ← Main component (form + result panel)
        └── index.css                ← All styles (single-page layout)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ and `pip3`
- Node.js 18+ and `npm`
- The trained model file: `backend/term_deposit_model.joblib`

### 1. Backend

```bash
cd backend

# Create and activate a virtual environment (required on macOS)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

> **macOS note:** Use `pip` (not `pip3`) after activating the venv. Do not use `pip3` at the system level — this causes the `externally-managed-environment` error.

> **sklearn version note:** The model was saved with scikit-learn `1.6.1`. If you see an `InconsistentVersionWarning` or `AttributeError: _RemainderColsList`, run:
> ```bash
> pip install "scikit-learn==1.6.1"
> ```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 📡 API Reference

### `GET /`
Health check. Returns whether the model is loaded.

**Response:**
```json
{ "status": "ok", "model_loaded": true }
```

---

### `POST /predict`

Run a prediction for a customer profile.

**Request body:**
```json
{
  "age": 58,
  "job": "retired",
  "marital": "single",
  "education": "tertiary",
  "default": "no",
  "balance": 8500,
  "housing": "no",
  "loan": "no",
  "contact": "cellular",
  "day": 20,
  "month": "oct",
  "campaign": 1,
  "never_contacted": false,
  "pdays": 10,
  "previous": 3,
  "poutcome": "success"
}
```

**Response:**
```json
{
  "prediction": "Likely to subscribe",
  "probability_percent": 82.0,
  "label": "yes"
}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | string | Human-readable verdict |
| `probability_percent` | float | Model's confidence (0–100) |
| `label` | string | `"yes"` or `"no"` — for frontend branching |

**Valid field values:**

| Field | Valid values |
|---|---|
| `job` | `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` |
| `marital` | `divorced`, `married`, `single` |
| `education` | `primary`, `secondary`, `tertiary`, `unknown` |
| `default`, `housing`, `loan` | `yes`, `no` |
| `contact` | `cellular`, `telephone`, `unknown` |
| `month` | `jan`, `feb`, `mar`, `apr`, `may`, `jun`, `jul`, `aug`, `sep`, `oct`, `nov`, `dec` |
| `poutcome` | `failure`, `other`, `success`, `unknown` |

---

## ☁️ Deployment

### Backend → Render

1. Push the `backend/` folder to GitHub
2. On [render.com](https://render.com): New → Web Service → connect your repo
3. Set **Root Directory** to `backend`
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable: `ALLOWED_ORIGIN` = your frontend URL (e.g., `https://your-app.vercel.app`)

> **Free tier note:** Render spins down idle services after ~15 minutes. The first request after idle takes 30–60 seconds to "cold start" — this is normal behaviour, not a bug.

### Frontend → Vercel

1. Push the `frontend/` folder to GitHub
2. On [vercel.com](https://vercel.com): New Project → import repo
3. Set **Root Directory** to `frontend`
4. Add environment variable: `VITE_API_URL` = your Render backend URL (e.g., `https://term-deposit-api.onrender.com`)
5. Deploy

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn `RandomForestClassifier` inside a `Pipeline` |
| Preprocessing | `ColumnTransformer` with `StandardScaler` + `OneHotEncoder` |
| Model serialisation | `joblib` |
| API Framework | FastAPI + Uvicorn |
| Data validation | Pydantic v2 |
| Frontend | React 18 + Vite |
| Styling | Vanilla CSS (no framework) |
| Dataset | [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) |

---

*Built for the AI/ML Entry-Level Shortlisting Task — UCI Bank Marketing dataset (Moro et al., 2014).*
