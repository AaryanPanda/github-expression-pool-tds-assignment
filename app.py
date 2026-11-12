"""
Expression Pool — shared training data for the TDS face-expression bonus question.

Students upload features.csv; the server stores it (SQLAlchemy), de-duplicates
people, retrains a linear-rank-v1 model after every upload, and serves the
latest model.json.

Run locally:   uvicorn app:app --reload
Environment:   DATABASE_URL  (default sqlite:///pool.db; Postgres URLs work)
               ADMIN_TOKEN   (needed to delete uploads; unset = deleting disabled)
"""
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

# Small cloud instances get a fraction of one CPU. Stop NumPy/SciPy from starting
# one thread per visible core, which makes them thrash. Must run before numpy loads.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

LABELS = ["neutral", "happy", "sad", "angry", "surprised", "contempt",
          "restrained-amusement", "scepticism", "uncertainty", "hopelessness"]
ID_COLS = {"photo_id", "intended_expression", "person", "person_id", "file", "filename", "path"}
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1]
MAX_UPLOAD_BYTES = 2_000_000


# ---------------------------------------------------------------- database
def _db_url() -> str:
    url = os.environ.get("DATABASE_URL", "sqlite:///pool.db")
    # Render/Heroku style URLs -> SQLAlchemy + psycopg 3
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


DB_URL = _db_url()
engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
)


class Base(DeclarativeBase):
    pass


def now():
    return datetime.now(timezone.utc)


class Upload(Base):
    __tablename__ = "uploads"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(60))
    filename: Mapped[str] = mapped_column(String(200))
    csv_text: Mapped[str] = mapped_column(Text)          # the original file, kept as-is
    columns: Mapped[list] = mapped_column(JSON)          # the 78 feature column names, in file order
    valid_people: Mapped[int] = mapped_column(Integer)
    new_people: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    people: Mapped[list["Person"]] = relationship(back_populates="upload", cascade="all, delete-orphan")


class Person(Base):
    """One person = 10 labelled photos. `hash` de-duplicates across uploads."""
    __tablename__ = "people"
    hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.id", ondelete="CASCADE"), index=True)
    labels: Mapped[list] = mapped_column(JSON)           # 10 label indices
    features: Mapped[list] = mapped_column(JSON)         # 10 x 78, in the upload's column order
    upload: Mapped[Upload] = relationship(back_populates="people")


class Model(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    people: Mapped[int] = mapped_column(Integer)
    accuracy: Mapped[float] = mapped_column(Float)       # cross-validated, unseen people
    best_c: Mapped[float] = mapped_column(Float)
    cv: Mapped[list] = mapped_column(JSON)               # [{C, acc}]
    per_expression: Mapped[list] = mapped_column(JSON)   # 10 accuracies
    weights: Mapped[list] = mapped_column(JSON)          # 10 x 156


Base.metadata.create_all(engine)


# ---------------------------------------------------------------- CSV parsing
def norm_label(s: str) -> str:
    s = str(s).strip().lower().replace("_", "-").replace(" ", "-")
    return s.replace("skepticism", "scepticism")


def to_float(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def photo_key(label, row, cols) -> str:
    """One photo, independent of person number, photo order and column order."""
    vals = [row[i] for i in sorted(range(len(cols)), key=lambda i: cols[i])]
    # Rounded to 4 decimals so the same photo still matches after Excel or other tools
    # re-save the numbers with fewer digits. Different real photos never agree on all 78.
    return str(label) + "|" + ",".join("nan" if v is None else f"{round(float(v), 4) + 0.0:.4f}" for v in vals)


def person_key(labels, feats, cols) -> str:
    """The same real person gets the same key even if renumbered and shuffled."""
    keys = sorted(photo_key(l, r, cols) for l, r in zip(labels, feats))
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def parse_csv(text: str):
    """Return (columns, people, skipped). Raises ValueError with a readable message."""
    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    if not rows:
        raise ValueError("The file is empty.")
    head = list(rows[0].keys())
    if "photo_id" not in head or "intended_expression" not in head:
        raise ValueError("Missing photo_id or intended_expression column. Is this the features.csv from the ZIP?")
    cols = [c for c in head if c not in ID_COLS]
    if len(cols) != 78:
        raise ValueError(f"Expected 78 feature columns, found {len(cols)}.")
    groups: dict[str, list] = {}
    for r in rows:
        pid = str(r["photo_id"]).strip()
        person = pid.rsplit("-", 1)[0] if "-" in pid else pid
        lab = norm_label(r["intended_expression"])
        groups.setdefault(person, []).append(
            (LABELS.index(lab) if lab in LABELS else -1, [to_float(r[c]) for c in cols]))
    people, skipped = [], 0
    for items in groups.values():
        labels = [l for l, _ in items]
        if len(items) != 10 or -1 in labels or len(set(labels)) != 10:
            skipped += 1
            continue
        feats = [[None if v is None else float(f"{v:.6g}") for v in x] for _, x in items]
        h = person_key(labels, feats, cols)
        people.append({"hash": h, "labels": labels, "features": feats})
    if not people:
        raise ValueError("No person had exactly 10 photos with the 10 different expressions.")
    return cols, people, skipped


# ---------------------------------------------------------------- training
def person_inputs(feats: np.ndarray) -> np.ndarray:
    """10x78 raw (NaN allowed) -> 10x156: per-person centred raw + within-person ranks in 0..1."""
    X = feats.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        ok = np.isfinite(col)
        col[~ok] = np.median(col[ok]) if ok.any() else 0.0
    ranks = np.apply_along_axis(lambda c: (rankdata(c) - 1) / (len(c) - 1), 0, X)
    return np.hstack([X - X.mean(axis=0), ranks])


def match_accuracy(W, X, y, groups, per_expr=None):
    correct = total = 0
    for start in range(0, len(y), 10):          # rows come in blocks of 10 = one person
        m = np.arange(start, start + 10)
        rows, cols = linear_sum_assignment(X[m] @ W.T, maximize=True)
        hits = cols == y[m][rows]
        correct += int(hits.sum())
        total += 10
        if per_expr is not None:
            for lab, hit in zip(y[m][rows], hits):
                per_expr[lab][0] += int(hit)
                per_expr[lab][1] += 1
    return correct / total


def fit(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    W = np.zeros((10, X.shape[1]))
    W[clf.classes_] = clf.coef_ / sc.scale_
    return W


def load_people(session: Session):
    """All stored people as (key, photo_keys, labels, feats_in_canon_order), duplicates removed."""
    rows = session.execute(select(Person.labels, Person.features, Upload.columns).join(Upload).order_by(Person.upload_id)).all()
    if not rows:
        return [], None
    canon = session.execute(select(Upload.columns).order_by(Upload.id)).scalars().first()
    # A stored person is a copy of an earlier one if it shares ANY photo with it (renumbered,
    # reshuffled or re-saved copies). Keep only the first of each real person.
    out, seen_photos = [], set()
    for labels, feats, cols in rows:
        if set(cols) != set(canon):
            continue
        pkeys = [photo_key(l, r, cols) for l, r in zip(labels, feats)]
        if any(k in seen_photos for k in pkeys):
            seen_photos.update(pkeys)
            continue
        seen_photos.update(pkeys)
        idx = [cols.index(c) for c in canon]
        canon_feats = [[r[k] for k in idx] for r in feats]
        out.append((person_key(labels, feats, cols), pkeys, labels, canon_feats))
    return out, canon


def unique_people_count() -> int:
    with Session(engine) as s:
        return len(load_people(s)[0])


def load_training_set(session: Session):
    people, _ = load_people(session)
    if not people:
        return None
    # load_people already merged copies, so every entry is a distinct real person
    # and each person is its own cross-validation group (exactly 10 photos).
    Xs, ys, gs = [], [], []
    for i, (_, _, labels, feats) in enumerate(people):
        arr = np.array([[np.nan if v is None else v for v in row] for row in feats], dtype=float)
        Xs.append(person_inputs(arr))
        ys.extend(labels)
        gs.extend([i] * 10)
    return np.vstack(Xs), np.array(ys), np.array(gs)


def train_and_store():
    with Session(engine) as s:
        data = load_training_set(s)
        if data is None:
            return
        X, y, groups = data
        n_people = len(np.unique(groups))
        if n_people < 5:
            return
        folds = list(GroupKFold(n_splits=min(5, n_people)).split(X, y, groups))
        cv, per_by_c = [], {}
        for C in CS:
            per = [[0, 0] for _ in range(10)]
            accs = [match_accuracy(fit(X[tr], y[tr], C), X[te], y[te], groups[te], per) for tr, te in folds]
            cv.append({"C": C, "acc": float(np.mean(accs))})
            per_by_c[C] = per
        best = max(cv, key=lambda r: r["acc"])
        W = fit(X, y, best["C"])
        assert W.shape == (10, 156) and np.isfinite(W).all()
        s.add(Model(people=n_people, accuracy=best["acc"], best_c=best["C"], cv=cv,
                    per_expression=[c / n if n else 0 for c, n in per_by_c[best["C"]]],
                    weights=[[float(f"{v:.6g}") for v in row] for row in W]))
        s.commit()


def model_is_behind() -> bool:
    with Session(engine) as s:
        people = len(load_people(s)[0])
        m = s.execute(select(Model.people).order_by(Model.id.desc()).limit(1)).scalar_one_or_none()
        return people >= 5 and (m is None or m != people)


_count_cache = {"n": None}


def _pool_count():
    if _count_cache["n"] is None:
        _count_cache["n"] = unique_people_count()
    return _count_cache["n"]


# One trainer at a time, in a SEPARATE PROCESS so the website keeps answering
# while it runs. Uploads during training trigger one more run afterwards.
_state = {"training": False, "dirty": False, "error": None, "since": None}
_lock = threading.Lock()
TRAIN_TIMEOUT_S = 1800


def _run_trainer_process():
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--train"],
                           capture_output=True, text=True, timeout=TRAIN_TIMEOUT_S)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["exit code %d" % r.returncode]
            _state["error"] = tail[0][:300]
        else:
            _state["error"] = None
    except subprocess.TimeoutExpired:
        _state["error"] = f"Training took longer than {TRAIN_TIMEOUT_S // 60} minutes and was stopped."
    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"


def request_training():
    with _lock:
        _state["dirty"] = True
        if _state["training"]:
            return
        _state["training"] = True
    try:
        while True:
            with _lock:
                if not _state["dirty"]:
                    return
                _state["dirty"] = False
                _state["since"] = time.time()
            _run_trainer_process()
    finally:
        with _lock:
            _state["training"] = False
            _state["since"] = None


def start_training_thread():
    threading.Thread(target=request_training, daemon=True).start()


# ---------------------------------------------------------------- API
app = FastAPI(title="Expression Pool")


@app.on_event("startup")
def catch_up():
    # After a restart (or if a previous run died), retrain when the model is behind the pool.
    if model_is_behind():
        start_training_thread()


STATIC = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/status")
def status():
    with Session(engine) as s:
        uploads = s.execute(select(Upload.id, Upload.name, Upload.filename, Upload.valid_people,
                                   Upload.new_people, Upload.created_at).order_by(Upload.id.desc())).all()
        people = _pool_count()
        m = s.execute(select(Model).order_by(Model.id.desc()).limit(1)).scalar_one_or_none()
        return {
            "people": people,
            "uploads": [{"id": u.id, "name": u.name, "filename": u.filename, "valid": u.valid_people,
                         "new": u.new_people, "at": u.created_at.isoformat()} for u in uploads],
            "training": _state["training"],
            "training_for": None if not _state["since"] else int(time.time() - _state["since"]),
            "error": _state["error"],
            "model": None if m is None else {
                "id": m.id, "at": m.created_at.isoformat(), "people": m.people, "accuracy": m.accuracy,
                "C": m.best_c, "cv": m.cv, "per_expression": m.per_expression},
        }


@app.post("/api/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...), name: str = Form("")):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large. features.csv should be well under 2 MB.")
    try:
        text = raw.decode("utf-8")
        cols, people, skipped = parse_csv(text)
    except UnicodeDecodeError:
        raise HTTPException(400, "This doesn't look like a CSV text file.")
    except ValueError as e:
        raise HTTPException(400, str(e))
    with Session(engine) as s:
        canon = s.execute(select(Upload.columns).order_by(Upload.id)).scalars().first()
        if canon and set(canon) != set(cols):
            raise HTTPException(400, "The feature columns don't match the pool. Use the features.csv from the course ZIP.")
        pool_photos = {k for _, pkeys, _, _ in load_people(s)[0] for k in pkeys}
        fresh, seen_photos = [], set()
        for p in people:
            pkeys = [photo_key(l, r, cols) for l, r in zip(p["labels"], p["features"])]
            if not any(k in pool_photos or k in seen_photos for k in pkeys):
                fresh.append(p)
            seen_photos.update(pkeys)
        up = Upload(name=(name.strip() or "Anonymous")[:60], filename=(file.filename or "features.csv")[:200],
                    csv_text=text, columns=cols, valid_people=len(people), new_people=len(fresh))
        up.people = [Person(**p) for p in fresh]
        s.add(up)
        s.commit()
    _count_cache["n"] = None
    if fresh:
        start_training_thread()
    return {"valid": len(people), "new": len(fresh), "duplicates": len(people) - len(fresh), "skipped": skipped}


@app.post("/api/test")
async def test_file(file: UploadFile = File(...)):
    """Score the latest model on a CSV WITHOUT saving it. Only people not already in the pool count."""
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large.")
    try:
        cols, people, skipped = parse_csv(raw.decode("utf-8"))
    except UnicodeDecodeError:
        raise HTTPException(400, "This doesn't look like a CSV text file.")
    except ValueError as e:
        raise HTTPException(400, str(e))
    with Session(engine) as s:
        m = s.execute(select(Model).order_by(Model.id.desc()).limit(1)).scalar_one_or_none()
        if m is None:
            raise HTTPException(404, "No model trained yet.")
        pool, canon = load_people(s)
        pool_keys = {k for k, _, _, _ in pool}
        pool_photos = {pk for _, pks, _, _ in pool for pk in pks}
        model = {"id": m.id, "people": m.people, "accuracy": m.accuracy}
        W = np.array(m.weights)
    if set(cols) != set(canon):
        raise HTTPException(400, "The feature columns don't match the pool.")
    idx = [cols.index(c) for c in canon]
    unseen = [p for p in people if p["hash"] not in pool_keys
              and not any(photo_key(l, r, cols) in pool_photos for l, r in zip(p["labels"], p["features"]))]
    correct = total = 0
    per = [[0, 0] for _ in range(10)]
    for p in unseen:
        X = np.array([[np.nan if r[k] is None else r[k] for k in idx] for r in p["features"]], dtype=float)
        for j in range(X.shape[1]):  # fill missing like training does
            ok = np.isfinite(X[:, j]); X[~ok, j] = np.median(X[ok, j]) if ok.any() else 0.0
        ranks = np.apply_along_axis(lambda c: (rankdata(c) - 1) / (len(c) - 1), 0, X)
        rows, picks = linear_sum_assignment(np.hstack([X, ranks]) @ W.T, maximize=True)
        y = np.array(p["labels"])
        for t, hit in zip(y[rows], picks == y[rows]):
            per[t][0] += int(hit); per[t][1] += 1
            correct += int(hit); total += 1
    return {"valid": len(people), "in_pool": len(people) - len(unseen), "unseen": len(unseen),
            "accuracy": (correct / total) if total else None,
            "per_expression": [(c / n) if n else None for c, n in per], "model": model}


@app.get("/api/export")
def export_all():
    """ZIP of every original upload plus one de-duplicated pooled CSV."""
    import re
    import zipfile
    buf = io.BytesIO()
    with Session(engine) as s, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        people, canon = load_people(s)
        if canon:
            out = io.StringIO()
            w = csv.writer(out)
            w.writerow(["photo_id", "intended_expression", *canon])
            for n, (_, _, labels, feats) in enumerate(people, start=1):
                for k, (lab, row) in enumerate(zip(labels, feats), start=1):
                    w.writerow([f"{n:04d}-{k:02d}", LABELS[lab], *["" if v is None else v for v in row]])
            z.writestr("pooled-features.csv", out.getvalue())
        uploads = s.execute(select(Upload.id, Upload.name, Upload.csv_text).order_by(Upload.id)).all()
        for uid, name, text in uploads:
            safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:30] or "upload"
            z.writestr(f"uploads/{uid:03d}-{safe}.csv", text)
        z.writestr("README.txt",
                   "Expression Pool export\n\n"
                   f"pooled-features.csv : {len(people)} unique people (duplicates across uploads removed),\n"
                   "                      10 photos each, 78 Py-Feat features, intended_expression label.\n"
                   f"uploads/            : the {len(uploads)} original features.csv files as uploaded.\n\n"
                   "Train with: python train_expressions.py pooled-features.csv\n")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="expression-pool-{stamp}.zip"'})


@app.get("/api/model.json")
def model_json(download: int = 1):
    with Session(engine) as s:
        m = s.execute(select(Model).order_by(Model.id.desc()).limit(1)).scalar_one_or_none()
        if m is None:
            raise HTTPException(404, "No model trained yet.")
        body = json.dumps({"version": "linear-rank-v1", "weights": m.weights}, separators=(",", ":"))
    headers = {"Content-Disposition": 'attachment; filename="model.json"'} if download else {}
    return Response(body, media_type="application/json", headers=headers)


@app.delete("/api/uploads/{upload_id}")
def delete_upload(upload_id: int, background: BackgroundTasks, x_admin_token: str = Header("")):
    token = os.environ.get("ADMIN_TOKEN")
    if not token or x_admin_token != token:
        raise HTTPException(403, "Admin token required.")
    with Session(engine) as s:
        up = s.get(Upload, upload_id)
        if up is None:
            raise HTTPException(404, "No such upload.")
        s.delete(up)
        s.commit()
    _count_cache["n"] = None
    start_training_thread()
    return JSONResponse({"deleted": upload_id})


if __name__ == "__main__" and "--train" in sys.argv:
    train_and_store()
