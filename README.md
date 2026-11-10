# Expression Pool

A shared training pool for the TDS Project 1 face-expression bonus question.
Classmates upload their `features.csv`; the server stores it in a SQL database
(SQLAlchemy), skips people already in the pool, retrains after every upload, and
serves the newest `model.json`.

## Files

| File | What it does |
|---|---|
| `app.py` | FastAPI server: database tables, CSV validation, training, API |
| `static/index.html` | The single page students use |
| `requirements.txt` | Python packages |

## Run on your machine (optional)

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000. Without `DATABASE_URL` it uses a local SQLite file `pool.db`.

## Deploy on Render

1. Push this folder to a new GitHub repo.
2. **Create a Postgres database** (free tiers exist on Render, Neon and Supabase; check their
   current limits). Copy its connection URL.
   Render's free web service has no permanent disk, so SQLite there would be wiped on every restart.
3. On Render: **New → Web Service**, pick the repo, then set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Environment variables:
     - `DATABASE_URL` = the Postgres URL (`postgres://…` or `postgresql://…` both work)
     - `ADMIN_TOKEN` = any long secret you choose
4. Share the Render URL with classmates.

Free Render services sleep when idle, so the first visit after a break can take ~30–60 seconds.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | The page |
| GET | `/api/status` | Pool size, uploads, latest model summary |
| POST | `/api/upload` | Form fields `file` (CSV) and `name` |
| GET | `/api/model.json` | Latest model (`?download=0` to view inline) |
| DELETE | `/api/uploads/{id}` | Remove a bad upload; header `X-Admin-Token` |

## Removing a bad upload

Open `https://<your-app>/?admin`, enter your `ADMIN_TOKEN`, and click **Delete** next to the upload.
The model retrains without it.

## Database tables

- `uploads`: one row per uploaded file (name, filename, the original CSV text, column names, counts)
- `people`: one row per person (10 photos), keyed by a hash so duplicates are stored once
- `models`: every trained model (accuracy, best C, per-expression accuracy, 10 × 156 weights)

## How the model is trained

156 inputs per photo: the 78 Py-Feat features (centred within each person) plus each feature's
rank among that person's 10 photos (0 to 1). A scikit-learn `LogisticRegression` is chosen by
5-fold cross-validation grouped by person, scored with the grader's one-to-one matching, then
refit on everyone. The coefficients are divided by `StandardScaler.scale_` so they apply to raw
inputs, as the question requires.
