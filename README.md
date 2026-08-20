# Term Deposit Subscription Predictor

A full-stack machine learning web app that predicts whether a bank customer will subscribe to a term deposit — built with a Random Forest classifier, a FastAPI backend, and a React + Vite frontend.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Project Structure](#project-structure)
- [How the Model Works](#how-the-model-works)
- [The Algorithm — Random Forest](#the-algorithm--random-forest)
- [Why Call Duration is Excluded](#why-call-duration-is-excluded)
- [The "Never Contacted" Checkbox Explained](#the-never-contacted-checkbox-explained)
- [Input Features](#input-features)
- [System Architecture](#system-architecture)
- [Data Flow — One Prediction End to End](#data-flow--one-prediction-end-to-end)
- [Getting Started Locally](#getting-started-locally)
- [API Reference](#api-reference)
- [Frontend Component Guide](#frontend-component-guide)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

---

## What This Project Does

Banks run phone campaigns to sell term deposits — fixed-interest savings accounts. Most customers say no. Calling the wrong people wastes agent time and annoys customers.

This tool gives a bank's marketing team a **pre-call score**: before picking up the phone, a manager enters a customer's profile and gets back:

- **A binary verdict** — "Likely to subscribe" or "Unlikely to subscribe"
- **A probability percentage** — e.g. 78.3% — to rank and prioritise

The model is trained on the **UCI Bank Marketing Dataset** (45,211 real campaign records from a Portuguese bank) and uses a **Random Forest Classifier** that was trained without call duration to avoid data leakage (more on this below).

---

## Project Structure

```
project 2/
│
├── README.md                        ← This file
│
├── backend/
│   ├── main.py                      ← FastAPI app: endpoints, schema, inference logic
│   ├── requirements.txt             ← Python dependencies (fastapi, sklearn, pandas, etc.)
│   ├── term_deposit_model.joblib    ← Trained sklearn Pipeline (binary file, not in git)
│   ├── venv/                        ← Python virtual environment (not in git)
│   └── README.md                    ← Backend-specific run & deploy instructions
│
└── frontend/
    ├── index.html                   ← HTML shell + Google Fonts (Inter) import
    ├── vite.config.js               ← Vite bundler config
    ├── package.json                 ← React 18, Vite 5 dependencies
    └── src/
        ├── main.jsx                 ← React entry point, mounts <App /> into #root
        ├── App.jsx                  ← Entire UI: form, state, fetch logic, result panel
        └── index.css                ← All styles — single-page 100vh layout, no framework
```

---

## How the Model Works

### The sklearn Pipeline

The trained model is a **scikit-learn `Pipeline`** — a single object that chains preprocessing and the classifier together. It is saved to disk with `joblib` as `term_deposit_model.joblib` (21 MB).

```
Customer Data (raw, with text fields)
            │
            ▼
┌──────────────────────────────────────────┐
│  ColumnTransformer                       │
│                                          │
│  Numerical columns (age, balance, etc.)  │
│    └─► StandardScaler                   │
│         Subtracts mean, divides by std   │
│         e.g. age=58 → normalised ~1.2   │
│                                          │
│  Categorical columns (job, month, etc.) │
│    └─► OneHotEncoder                    │
│         Turns text into binary columns  │
│         e.g. job="retired" →            │
│         [0,0,0,0,0,0,0,0,1,0,0,0]      │
└──────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│  RandomForestClassifier                  │
│  → outputs predict_proba()               │
│    e.g. [0.18, 0.82] → P(yes) = 82%    │
└──────────────────────────────────────────┘
            │
            ▼
     { label: "yes", probability: 82.0 }
```

Because preprocessing is baked into the Pipeline, the API can receive the raw JSON payload (with text like `"retired"`, `"oct"`) and pass it straight into `model.predict_proba()` — no manual encoding needed at inference time.

### Startup Loading

When the FastAPI server starts, `load_model()` runs once (decorated with `@app.on_event("startup")`). It loads the `.joblib` file into a global `model` variable. If the file is missing, the server still starts — it just returns an HTTP 503 on any `/predict` call with a clear error message, which makes Render deploy logs easy to read.

---

## The Algorithm — Random Forest

### Step 1: Decision Trees

A **Decision Tree** makes predictions by asking a series of yes/no questions about the features:

```
                    Was previous campaign successful?
                          /                \
                        YES                 NO
                        /                    \
               Balance > 5000?           Age > 55?
               /           \             /        \
             YES             NO        YES          NO
              |               |         |            |
           LIKELY          LIKELY    LIKELY       UNLIKELY
```

Each node picks the feature and threshold that best separates subscribers from non-subscribers (measured by Gini impurity or entropy). The tree keeps splitting until it reaches a leaf — the final prediction.

**The problem:** A single deep tree memorises the training data perfectly but fails on new customers (overfitting).

### Step 2: The Forest

A **Random Forest** solves this by building many trees (typically 100–500) and combining their votes. Each tree is made deliberately different through two sources of randomness:

1. **Bootstrap sampling** — each tree trains on a random sample of rows (drawn with replacement). About 37% of rows are left out of each tree.
2. **Feature subsampling** — at every split, the tree only considers a random subset of the 16 features, not all of them. This prevents all trees from making the same splits.

```
Customer Data
      │
      ├──► Tree 001 → "Likely"    ─┐
      ├──► Tree 002 → "Unlikely"   │
      ├──► Tree 003 → "Likely"     │  majority
      ├──► Tree 004 → "Likely"     ├─ vote
      │         ...                │
      └──► Tree 100 → "Likely"   ─┘
                                   │
                Final prediction: "LIKELY"
                Probability: 74%   (74/100 trees voted yes)
```

Because each tree sees different data and different features, their errors are uncorrelated — individual mistakes cancel out across the ensemble.

### Why Random Forest for This Problem

| Property | Why it matters here |
|---|---|
| Handles mixed data | Inputs include both numbers (age, balance) and text (job, month) |
| Non-linear decisions | Subscription likelihood isn't a straight line — trees capture complex boundaries |
| Robust to outliers | Some customers have extreme balances; threshold-based splits are unaffected |
| No strong assumptions | Unlike logistic regression, no linearity or normality assumption needed |
| Probability output | `predict_proba()` gives a confidence score, not just a binary label |

---

## Why Call Duration is Excluded

The UCI dataset includes a `duration` column — the length of the last call in seconds. Analysis shows it is the single most predictive feature: longer calls very strongly predict subscription.

**But it cannot be used.** Duration is only known *after* the call ends. By that point the outcome is already known too (if someone stayed on the phone for 10 minutes, they almost certainly said yes). Including it would mean the model "cheats" — it would learn from future information.

This is called **data leakage**, and it produces models that score brilliantly on paper but are useless in production because the feature doesn't exist at decision time.

This model is designed for a **pre-call decision**: should we call this customer at all? So `duration` is excluded from both training and inference.

---

## The "Never Contacted" Checkbox Explained

The UCI dataset has a `pdays` feature — the number of days since the customer was last contacted in a *previous* campaign. But if the customer was never contacted before, `pdays` is stored as `-1` in the dataset as a sentinel value.

The checkbox in the UI handles this cleanly:

```
☑ Never contacted before this campaign
  └─ pdays sent as -1 to model
     was_contacted_before = 0

☐ (unchecked — was contacted before)
  └─ shows "Days Since Last Contact" field
     pdays = whatever the user enters
     was_contacted_before = 1
```

From `backend/main.py` lines 93–94:
```python
pdays_value = -1 if customer.never_contacted else customer.pdays
was_contacted_before = 0 if customer.never_contacted else 1
```

`was_contacted_before` is an **engineered feature** added to make the binary distinction explicit for the model (rather than relying on it to learn that `-1` is special).

**Why it matters:** Whether a customer was contacted before — and what happened — is one of the strongest predictors. A `poutcome = "success"` (they subscribed last time) is a very strong positive signal. A brand-new customer has no history, which generally lowers confidence.

---

## Input Features

The model receives **16 features** (15 from the form + 1 engineered in the backend):

### Customer Profile

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `age` | int | 18–100 | Customer's age |
| `job` | string | `admin.` `blue-collar` `entrepreneur` `housemaid` `management` `retired` `self-employed` `services` `student` `technician` `unemployed` `unknown` | Employment type |
| `marital` | string | `married` `divorced` `single` | Marital status |
| `education` | string | `primary` `secondary` `tertiary` `unknown` | Highest qualification |

### Financial Status

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `default` | string | `yes` `no` | Has credit in default? |
| `balance` | int | any integer (can be negative) | Average yearly account balance in € |
| `housing` | string | `yes` `no` | Has a housing loan? |
| `loan` | string | `yes` `no` | Has a personal loan? |

### Current Campaign

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `contact` | string | `cellular` `telephone` `unknown` | How the customer was reached |
| `day` | int | 1–31 | Day of month of last contact |
| `month` | string | `jan`–`dec` | Month of last contact |
| `campaign` | int | ≥ 1 | Number of times contacted in this campaign |

### Previous Campaign

| Field | Type | Valid values | Notes |
|---|---|---|---|
| `never_contacted` | bool | `true` / `false` | UI-only flag — controls `pdays` and `was_contacted_before` |
| `pdays` | int | ≥ 1 (or -1 if never contacted) | Days since previous campaign contact |
| `previous` | int | ≥ 0 | Number of contacts before this campaign |
| `poutcome` | string | `unknown` `failure` `other` `success` | Previous campaign outcome |

### Engineered in Backend

| Feature | How it's set | Purpose |
|---|---|---|
| `was_contacted_before` | `0` if `never_contacted=true`, `1` otherwise | Makes the "never contacted" status an explicit binary signal for the model |

---

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Browser (React)                    │
│                                                       │
│  App.jsx — single component, useState for:            │
│    form values, result, loading, error, probWidth     │
│                                                       │
│  handleSubmit() → coerces strings to numbers →       │
│  fetch POST /predict → updates result state          │
│                                                       │
│  index.css — 100vh flex layout, form scrolls         │
│  internally, result card stays fixed beside it        │
└──────────────────────┬───────────────────────────────┘
                       │  HTTP POST /predict
                       │  Content-Type: application/json
                       ▼
┌──────────────────────────────────────────────────────┐
│              FastAPI (Python, Uvicorn)                │
│                                                       │
│  CORS middleware — allows any origin in dev;         │
│  in prod, ALLOWED_ORIGIN env var locks it down       │
│                                                       │
│  POST /predict                                        │
│    1. Pydantic (CustomerInput) validates JSON        │
│    2. Builds pdays_value & was_contacted_before      │
│    3. Creates a 1-row pandas DataFrame               │
│    4. model.predict_proba(row)[0][1] → P(yes)        │
│    5. Returns PredictionResponse JSON                │
│                                                       │
│  GET / → health check { status, model_loaded }       │
└──────────────────────┬───────────────────────────────┘
                       │  loaded once on startup
                       ▼
              term_deposit_model.joblib
              (sklearn Pipeline: ColumnTransformer
               → StandardScaler + OneHotEncoder
               → RandomForestClassifier)
```

---

## Data Flow — One Prediction End to End

```
1. User fills the form (or clicks "Load likely example")
   Form state: { age:58, job:"retired", balance:8500,
                 poutcome:"success", never_contacted:false,
                 pdays:10, previous:3, ... }

2. handleSubmit() coerces string inputs to numbers:
   payload = { age:58, balance:8500, day:20,
               campaign:1, pdays:10, previous:3,
               never_contacted:false, ... }

3. fetch("POST /predict", JSON.stringify(payload))

4. FastAPI receives → Pydantic validates field types and ranges
   (age must be int 18-100, campaign must be ≥1, etc.)

5. Backend computes engineered features:
   pdays_value = 10          (never_contacted=false → use actual value)
   was_contacted_before = 1

6. pandas DataFrame created (1 row, 16 columns)

7. Pipeline runs:
   a. ColumnTransformer:
      - StandardScaler normalises age, balance, day, campaign, pdays, previous
      - OneHotEncoder encodes job, marital, education, default, housing,
        loan, contact, month, poutcome
   b. RandomForestClassifier.predict_proba(row)
      → e.g. [0.18, 0.82]  (18% no, 82% yes)

8. label = "yes"  (proba >= 0.5)
   prediction = "Likely to subscribe"
   probability_percent = 82.0

9. Response JSON:
   { "prediction": "Likely to subscribe",
     "probability_percent": 82.0,
     "label": "yes" }

10. React sets result state → renders:
    - Result row: "Likely to subscribe"
    - Probability bar animates from 0% → 82% (CSS transition)
    - Verdict badge: green "✓ Likely to subscribe" (fade-up animation)
```

---

## Getting Started Locally

### Prerequisites

- Python 3.9+ (Python 3.14 works — tested)
- Node.js 18+ and npm
- `backend/term_deposit_model.joblib` — the trained model file from your notebook

### Backend

```bash
cd backend

# Create a virtual environment (required on macOS — avoids externally-managed-environment error)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If sklearn version mismatch error appears (model was saved with 1.6.1):
pip install "scikit-learn==1.6.1"

# Start the server
uvicorn main:app --reload --port 8000
```

- API: `http://localhost:8000`
- Swagger UI (interactive docs): `http://localhost:8000/docs`

> **Note:** The `--reload` flag hot-reloads `main.py` on save. Remove it in production.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

- App: `http://localhost:5173`

The frontend reads `VITE_API_URL` from environment — defaults to `http://localhost:8000` if not set. The `.env.example` file in `frontend/` shows the format.

---

## API Reference

### `GET /`

Health check.

```json
{ "status": "ok", "model_loaded": true }
```

Returns `"model_loaded": false` if `term_deposit_model.joblib` is missing (server still runs, `/predict` returns 503).

---

### `POST /predict`

Predict subscription likelihood for one customer.

**Request:**
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

**Error (model missing):** HTTP 503
```json
{ "detail": "Model not loaded. Make sure term_deposit_model.joblib is present next to main.py." }
```

---

## Frontend Component Guide

The entire frontend lives in two files.

### `src/App.jsx`

One default export `App()` with this state:

| State | Type | Purpose |
|---|---|---|
| `form` | object | All 16 field values, initialised to `initialForm` defaults |
| `result` | object / null | API response: `{ prediction, probability_percent, label }` |
| `loading` | bool | Shows spinner, disables submit button |
| `error` | string / null | Shown as red error box below the form |
| `probWidth` | number | Drives the CSS width of the probability bar (set 60ms after result arrives to trigger the animation) |

Two helper components:
- `Field({ label, children, full })` — a labelled form field wrapper
- `Sel({ value, onChange, options })` — a controlled `<select>` element

Two example presets:
- `LIKELY_EXAMPLE` — 58yo retired, high balance, previous success, October contact
- `UNLIKELY_EXAMPLE` — 28yo blue-collar, negative balance, all loans, never contacted

### `src/index.css`

Single-page layout using CSS `100vh`:
- `html, body, #root` — `height: 100%`, `overflow: hidden`
- `.page` — `display: flex; flex-direction: column; height: 100vh`
- `.ledger-grid` — `flex: 1; min-height: 0` — fills remaining space
- `.form-card` — `overflow-y: auto` — only the form scrolls internally
- `.result-card` — `position: sticky; top: 24px` — always visible

On mobile (≤ 720px): reverts to normal page scroll, cards stack vertically.

---

## Deployment

### Backend → Render (free tier)

1. Push `backend/` to a GitHub repo
2. Render → New → Web Service → connect repo
3. **Root Directory:** `backend`
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. **Environment variable:** `ALLOWED_ORIGIN` = your Vercel frontend URL

> Free tier spins down after ~15 minutes of inactivity. First request after idle takes 30–60 seconds (cold start). This is normal.

### Frontend → Vercel (recommended)

1. Push `frontend/` to GitHub
2. Vercel → New Project → import repo
3. **Root Directory:** `frontend`
4. **Environment variable:** `VITE_API_URL` = your Render backend URL
5. Deploy

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| ML Algorithm | `RandomForestClassifier` (scikit-learn) | 1.6.1 |
| Preprocessing | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder` | 1.6.1 |
| Model serialisation | `joblib` | 1.5.x |
| API framework | FastAPI | 0.141.x |
| ASGI server | Uvicorn (with standard extras) | 0.52.x |
| Data validation | Pydantic v2 | 2.x |
| Data manipulation | pandas | 3.x |
| Frontend framework | React | 18.2 |
| Bundler | Vite | 5.1 |
| Styling | Vanilla CSS (no framework) | — |
| Font | Inter (Google Fonts) | — |
| Dataset | UCI Bank Marketing Dataset | 2014 |

---

*UCI Bank Marketing Dataset: Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22–31.*
