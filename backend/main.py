"""FastAPI server (ONLINE phase).

Run from the project root:
    uvicorn backend.main:app --port 8000
then open http://127.0.0.1:8000
Set CBIR_ARTIFACTS_DIR to serve another index (e.g. artifacts_test).
"""
import json
import random
import time
from contextlib import closing

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from . import config
from .db import database
from .db.fuzzy import TagMatcher
from .engine import SearchEngine, SearchError
from .schemas import DescriptorSpec, QueryInfo, ResultItem, SearchRequest, SearchResponse

engine = SearchEngine(config.ARTIFACTS_DIR)
app = FastAPI(title="CBIR - MIRFLICKR-25k", version="1.0")

with closing(engine.connect()) as _conn:
    _URLS = {row["id"]: row["url"] for row in _conn.execute("SELECT id, url FROM images")}
    tag_matcher = TagMatcher(_conn)


@app.get("/api/config")
def get_config():
    return {
        "n_images": len(engine.ids),
        "descriptors": {
            name: {**info, "default_metric": info["metrics"][0]}
            for name, info in engine.descriptors.items()
        },
        "params": engine.params,
        "build_seconds": engine.stats["build_seconds"],
    }


@app.get("/api/random")
def random_images(n: int = 24):
    ids = random.sample(engine.ids.tolist(), min(n, len(engine.ids)))
    return [{"image_id": i, "url": _url(i)} for i in ids]


@app.get("/api/images/{image_id}")
def image_info(image_id: int):
    with closing(engine.connect()) as conn:
        info = database.get_image(conn, image_id)
    if info is None:
        raise HTTPException(404, f"unknown image id {image_id}")
    return info


@app.get("/api/tags/suggest")
def suggest(prefix: str, limit: int = 10):
    with closing(engine.connect()) as conn:
        return database.suggest_tags(conn, prefix, limit)


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    return _run_search(req)


@app.post("/api/search/upload", response_model=SearchResponse)
async def search_upload(file: UploadFile = File(...), params: str = Form("{}")):
    try:
        req = SearchRequest(**json.loads(params))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(422, f"invalid params: {exc}")
    data = await file.read()
    return _run_search(req, upload=data)


# --------------------------------------------------------------------- logic
def _url(image_id):
    return _URLS.get(image_id, "")


def _run_search(req: SearchRequest, upload: bytes = None):
    keywords = [w.strip() for w in req.keywords if w.strip()]
    fuzz = req.fuzziness / 100.0
    query = QueryInfo(kind="keywords", keywords=keywords, keyword_mode=req.keyword_mode)
    common = {"fuzziness": req.fuzziness, "k_requested": req.k, "index_requested": req.index,
              "n_images": len(engine.ids)}
    conn = engine.connect()
    try:
        # Keywords: expand each one to close tags (typo tolerance), then match in SQLite.
        t0 = time.perf_counter()
        expansions, keyword_hits = {}, None
        if keywords:
            groups = [tag_matcher.expand(w, fuzz) for w in keywords]
            expansions = {w: list(g) for w, g in zip(keywords, groups)}
            keyword_hits = database.keyword_search(conn, groups, req.keyword_mode)
        keyword_ms = (time.perf_counter() - t0) * 1000
        common.update(keyword_expansions=expansions,
                      keyword_matches=len(keyword_hits) if keyword_hits is not None else None)

        # Keyword-only query: rank by number of matched keywords.
        if upload is None and req.image_id is None:
            if not keywords:
                raise HTTPException(400, "give keywords, an image id or an uploaded image")
            hits = keyword_hits[:req.k]
            tags = database.get_tags(conn, [i for i, _ in hits])
            results = [ResultItem(image_id=i, url=_url(i), rank=r + 1, matched_keywords=n, tags=tags[i])
                       for r, (i, n) in enumerate(hits)]
            return SearchResponse(query=query, results=results, descriptors=[], index_used="SQLite tag index",
                                  time_ms=keyword_ms, results_returned=len(results), **common)

        # Image query (database image / previous result, or upload).
        try:
            specs_in = [(d.name, d.metric, d.weight) for d in req.descriptors] or [("color", None, 1.0)]
            if upload is not None:
                specs = engine.normalize_specs(specs_in, engine.upload_descriptors)
                t_desc = time.perf_counter()
                qvecs, exclude = engine.vectors_for_upload(upload, [n for n, _, _ in specs]), None
                describe_ms = (time.perf_counter() - t_desc) * 1000
                query.kind = "upload"
            else:
                specs = engine.normalize_specs(specs_in, engine.descriptors)
                qvecs, exclude = engine.vectors_for_image_id(req.image_id)
                describe_ms = 0.0
                query.kind, query.image_id, query.url = "database", req.image_id, _url(req.image_id)
        except SearchError as exc:
            raise HTTPException(400, str(exc))
        used = [DescriptorSpec(name=n, metric=m, weight=w) for n, m, w in specs]

        rows = None
        if keyword_hits is not None:
            rows = [engine.row_of[i] for i, _ in keyword_hits if i in engine.row_of]
            if not rows:
                return SearchResponse(query=query, results=[], descriptors=used, index_used="SQLite tag index",
                                      time_ms=keyword_ms, results_returned=0, **common)
        res_rows, res_d, info = engine.search(qvecs, specs, req.k, req.index, rows=rows, exclude_row=exclude)
        keep, rel = engine.apply_fuzziness(res_d, fuzz)
        res_rows, res_d, rel = res_rows[keep], res_d[keep], rel[keep]

        ids = [int(engine.ids[r]) for r in res_rows]
        details = engine.per_descriptor_distances(qvecs, specs, res_rows)
        tags = database.get_tags(conn, ids)
        results = [ResultItem(image_id=i, url=_url(i), rank=r + 1, distance=float(d), relative_distance=float(rr),
                              distances={key: float(v[r]) for key, v in details.items()}, tags=tags[i])
                   for r, (i, d, rr) in enumerate(zip(ids, res_d, rel))]
        return SearchResponse(
            query=query, results=results, descriptors=used, index_used=info["index_used"],
            time_ms=info["time_ms"] + (keyword_ms if keywords else 0) + describe_ms,
            distance_computations=info["distance_computations"], results_returned=len(results), **common)
    finally:
        conn.close()


# Static files: images from disk, then the web client (mounted last, at the root).
app.mount("/images", StaticFiles(directory=engine.images_dir), name="images")
app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
