# Frontend — Term Deposit Ledger UI

React + Vite app that calls the FastAPI backend's `/predict` endpoint.

## Run locally

```bash
npm install
cp .env.example .env      # edit VITE_API_URL if backend runs elsewhere
npm run dev
```

Opens at http://localhost:5173. Make sure the backend is running at the URL
set in `.env` (defaults to http://localhost:8000).

## Deploy

The frontend is plain static files after build, so it does **not** need
Render — deploy it free on **Vercel** (recommended, fastest) or as a Render
Static Site.

### Option A — Vercel (recommended)
1. Push this `frontend/` folder to GitHub.
2. Go to vercel.com → New Project → import your repo.
3. Set **Root Directory** to `frontend`.
4. Add environment variable `VITE_API_URL` = your Render backend URL
   (e.g. `https://term-deposit-api.onrender.com`).
5. Deploy. Vercel builds with `npm run build` automatically.

### Option B — Render Static Site
1. New → Static Site → connect repo → Root Directory: `frontend`.
2. **Build Command:** `npm install && npm run build`
3. **Publish Directory:** `dist`
4. Add environment variable `VITE_API_URL` pointing to your backend service URL.

Once both are deployed, go back to your **backend's** Render settings and set
`ALLOWED_ORIGIN` to this frontend's URL, so CORS only allows your own app.
