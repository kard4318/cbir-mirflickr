"""SQLite storage for image paths, tags, metadata and ground-truth labels.

The images themselves stay on disk; the database only stores their relative path
and URL. Descriptors are stored in .npy files next to the database (see
scripts/build_index.py), not in SQLite.
"""
import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id        INTEGER PRIMARY KEY,
    filename  TEXT NOT NULL,
    url       TEXT NOT NULL,
    width     INTEGER,
    height    INTEGER,
    title     TEXT,
    owner     TEXT,
    license   TEXT,
    camera    TEXT,
    date_taken TEXT,
    exif_json TEXT
);
CREATE TABLE IF NOT EXISTS tags (
    image_id INTEGER NOT NULL REFERENCES images(id),
    tag      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_tags_image ON tags(image_id);
CREATE TABLE IF NOT EXISTS labels (
    image_id INTEGER NOT NULL REFERENCES images(id),
    label    TEXT NOT NULL,   -- e.g. 'dog'
    kind     TEXT NOT NULL    -- 'potential' (dog.txt) or 'relevant' (dog_r1.txt)
);
CREATE INDEX IF NOT EXISTS idx_labels_image ON labels(image_id);
"""


def connect(path):
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def create(path):
    if path.exists():
        path.unlink()
    conn = connect(path)
    conn.executescript(SCHEMA)
    return conn


def insert_images(conn, records):
    """records: dicts with id, filename, url, width, height, tags, exif, license."""
    img_rows, tag_rows = [], []
    for r in records:
        exif, lic = r.get("exif", {}), r.get("license", {})
        camera = " ".join(x for x in (exif.get("Make"), exif.get("Model")) if x) or None
        date = exif.get("Date and Time (Original)") or exif.get("Date and Time")
        img_rows.append((r["id"], r["filename"], r["url"], r.get("width"), r.get("height"),
                         lic.get("Picture title") or None, lic.get("Owner username") or None,
                         lic.get("License") or None, camera, date,
                         json.dumps(exif, ensure_ascii=False) if exif else None))
        tag_rows.extend((r["id"], t) for t in r.get("tags", []))
    conn.executemany("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?)", img_rows)
    conn.executemany("INSERT INTO tags VALUES (?,?)", tag_rows)
    conn.commit()


def insert_labels(conn, labels, valid_ids):
    rows = []
    for name, ids in labels.items():
        kind = "relevant" if name.endswith("_r1") else "potential"
        label = name[:-3] if kind == "relevant" else name
        rows.extend((i, label, kind) for i in ids if i in valid_ids)
    conn.executemany("INSERT INTO labels VALUES (?,?,?)", rows)
    conn.commit()


# ------------------------------------------------------------------ queries
def normalize_keyword(word):
    """Match the cleaned tag format: lowercase, no spaces."""
    return "".join(word.lower().split())


def keyword_search(conn, groups, mode="or"):
    """Search images by keyword groups.

    groups: one {tag: similarity} dict per keyword (the keyword's exact tag, plus
    close tags when typo tolerance is on). An image matches a keyword when it
    carries any tag of that keyword's group.
    mode="and": every keyword must match; mode="or": at least one.
    Returns [(image_id, n_matched_keywords)] ordered by matched keywords, then
    summed similarity (exact matches first), then id.
    """
    groups = [g for g in groups if g] if mode == "or" else groups
    if not groups or any(not g for g in groups):
        return []
    all_tags = sorted({t for g in groups for t in g})
    scores = {}  # image_id -> best similarity per keyword group
    for start in range(0, len(all_tags), 900):  # SQLite parameter limit
        chunk = all_tags[start:start + 900]
        sql = f"SELECT image_id, tag FROM tags WHERE tag IN ({','.join('?' * len(chunk))})"
        for row in conn.execute(sql, chunk):
            best = scores.setdefault(row["image_id"], [0.0] * len(groups))
            for gi, g in enumerate(groups):
                if row["tag"] in g:
                    best[gi] = max(best[gi], g[row["tag"]])
    hits = []
    for image_id, best in scores.items():
        n = sum(s > 0 for s in best)
        if mode == "and" and n < len(groups):
            continue
        hits.append((-n, -sum(best), image_id))
    hits.sort()
    return [(image_id, -neg_n) for neg_n, _, image_id in hits]


def suggest_tags(conn, prefix, limit=10):
    prefix = normalize_keyword(prefix)
    if not prefix:
        return []
    sql = ("SELECT tag, COUNT(*) AS n FROM tags WHERE tag LIKE ? GROUP BY tag "
           "ORDER BY n DESC LIMIT ?")
    return [{"tag": r["tag"], "count": r["n"]} for r in conn.execute(sql, (prefix + "%", limit))]


def get_tags(conn, image_ids):
    """{image_id: [tags]} for the given ids."""
    ids = list(image_ids)
    out = {i: [] for i in ids}
    for start in range(0, len(ids), 900):  # SQLite parameter limit
        chunk = ids[start:start + 900]
        sql = f"SELECT image_id, tag FROM tags WHERE image_id IN ({','.join('?' * len(chunk))})"
        for row in conn.execute(sql, chunk):
            out[row["image_id"]].append(row["tag"])
    return out


def get_image(conn, image_id):
    row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if row is None:
        return None
    info = dict(row)
    info["exif"] = json.loads(info.pop("exif_json") or "{}")
    info["tags"] = get_tags(conn, [image_id])[image_id]
    info["labels"] = [r["label"] for r in conn.execute(
        "SELECT DISTINCT label FROM labels WHERE image_id = ? AND kind = 'potential' ORDER BY label",
        (image_id,))]
    return info


def get_labels(conn, kind="potential"):
    """{image_id: set(labels)} for the evaluation ground truth."""
    out = {}
    for row in conn.execute("SELECT image_id, label FROM labels WHERE kind = ?", (kind,)):
        out.setdefault(row["image_id"], set()).add(row["label"])
    return out
