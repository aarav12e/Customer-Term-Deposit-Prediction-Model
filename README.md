# Term Deposit Predictor — Full-Stack Demo

A FastAPI backend (serving your trained sklearn model) + React frontend
(a bank-ledger-styled intake form) for the Customer Term Deposit Prediction
ML task.

```
project/
├── backend/          FastAPI app — put term_deposit_model.joblib here
│   ├── main.py
│   ├── requirements.txt
│   └── README.md      (backend-specific run/deploy steps)
└── frontend/          React + Vite app
    ├── src/
    ├── package.json
    └── README.md       (frontend-specific run/deploy steps)
```

## Quick start (local)

**Terminal 1 — backend:**
```bash
cd backend
# copy your term_deposit_model.joblib into this folder first
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Terminal 2 — frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — fill in a customer profile, submit, and you'll
see a prediction with an animated "stamp" verdict.

## Deploying for real

1. **Backend → Render.** See `backend/README.md`. Free tier, spins down
   after ~15 min idle (30-60s cold start on next request — normal, not a bug).
2. **Frontend → Vercel** (recommended) or Render Static Site. See
   `frontend/README.md`. Set `VITE_API_URL` to your deployed backend's URL.
3. Once both are live, set `ALLOWED_ORIGIN` on the backend to your frontend's
   URL so CORS is locked down to your own app instead of `*`.

## Where the model file comes from

`backend/term_deposit_model.joblib` is produced by Section 13 of your
Colab notebook ("Save Model for Deployment") — it's the same file that gets
auto-downloaded when you run that cell. Copy it into `backend/` before
running or deploying.
