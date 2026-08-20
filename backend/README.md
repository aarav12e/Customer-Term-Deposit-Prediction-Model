# Backend — Term Deposit Predictor API

FastAPI service that wraps the trained sklearn pipeline from the notebook.

## Before running

Copy your trained model file into this folder, named exactly:

```
backend/term_deposit_model.joblib
```

This is the file your Colab notebook downloads at the end (Section 13:
"Save Model for Deployment"). Without it, `/predict` will return a 503 error.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000/docs to test the `/predict` endpoint directly.

## Deploy on Render

1. Push this `backend/` folder to a GitHub repo (as a subfolder, or its own repo).
2. On [render.com](https://render.com) → New → Web Service → connect your repo.
3. Set **Root Directory** to `backend` (if it's a subfolder of a bigger repo).
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add an environment variable `ALLOWED_ORIGIN` set to your deployed frontend's URL
   once you have it (e.g. `https://your-app.vercel.app`), so CORS only allows your
   own frontend to call the API.
7. Deploy. Render gives you a URL like `https://term-deposit-api.onrender.com`.

**Note:** Render's free tier spins the service down after ~15 minutes of
inactivity. The first request after that will take 30-60 seconds to "wake up" —
this is normal, not a bug.
