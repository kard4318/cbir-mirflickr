# Report guide: CBIR search engine on MIRFLICKR-25k

This document explains the whole project and proposes a chapter-by-chapter plan
for the report. For each chapter you will find:

- **What to explain**: the content, already written so that it can be adapted directly,
- **Numbers**: the exact measured values (all from `report_assets/` or the logs of the runs),
- **Visuals**: what to insert, marked like this:
  - `[VIDEO]`: a screenshot to take from the demo video,
  - `[FIGURE]`: a ready-made figure in `report_assets/`,
  - `[TABLE]`: a table to reproduce from this guide.

Course: *Indexation et recherche dans des bases massives multimédia* (SUP'COM, M. Tébourbi).

---

## Suggested report outline

| # | Chapter | Length (indicative) |
|---|---|---|
| 0 | Abstract | ½ page |
| 1 | Introduction: context, objectives, constraints | 1 page |
| 2 | The dataset | 1 page |
| 3 | System architecture | 1–2 pages |
| 4 | Visual descriptors | 3 pages |
| 5 | Similarity measures and descriptor combination | 1 page |
| 6 | Indexing: VP-Tree and brute force | 2 pages |
| 7 | Text search and fuzziness | 1 page |
| 8 | The web application (API + interface) | 2 pages |
| 9 | Evaluation: retrieval quality | 2 pages |
| 10 | Evaluation: speed, VP-Tree vs brute force | 2 pages |
| 11 | Implementation challenges | 1 page |
| 12 | Limitations and future work | ½ page |
| 13 | Conclusion | ½ page |
| A | Appendix: installation, commands, API, project structure | 1 page |

---

## 0. Abstract (write it last)

> We built a content-based image retrieval engine for the 25,000 images of
> MIRFLICKR-25k without using Elasticsearch. Each image is described by six
> visual descriptors: four computed by us (HSV color histogram, LBP, HOG and
> VGG16 deep features) and two provided MPEG-7 descriptors (Edge Histogram and
> Homogeneous Texture). The descriptors are indexed in our own VP-Tree
> implementation, compared with brute-force search. A FastAPI server answers
> queries by image (database or uploaded) and/or by keywords (Flickr tags in
> SQLite, with typo tolerance) and returns JSON to a web client. On 100 random
> queries, VGG16 reaches a precision at 10 of 0.892, against 0.65–0.70 for the
> best hand-crafted descriptors and 0.564 for a random ranking. The VP-Tree is
> exact and skips up to 79 % of the distance computations for low-dimensional
> descriptors, but cannot prune in high dimension (HOG, VGG16), which illustrates
> the curse of dimensionality.

---

## 1. Introduction

**What to explain**

- **Context.** Large image collections cannot be searched by file name. CBIR
  (Content-Based Image Retrieval) describes each image by numerical vectors
  ("descriptors") and finds the images whose vectors are closest to the
  query's vector. The two challenges are **quality** (does the descriptor
  capture what the user means by "similar"?) and **speed** (how to avoid
  comparing the query with every image?).
- **Objectives of the project**
  1. Search by keywords AND/OR by an image (database image, uploaded image, or a previous result).
  2. Let the user choose one or several descriptors and combine them with weights.
  3. Implement our **own index structure** (a VP-Tree) and compare it with brute force.
  4. Provide a web client and a JSON server, and show the search time and the index used.
  5. Evaluate quality (precision, recall, mAP) and speed.
- **Imposed constraints**
  - No Elasticsearch / Kibana / Logstash: the index is written by us.
  - Backend: **FastAPI** with JSON. Frontend: plain **HTML/CSS/JavaScript**. Database: **SQLite**.
  - Images stay in folders on disk. The database stores paths, tags and metadata.
    Descriptors are stored in files loaded when the server starts.
  - Two phases, as in the course: **offline** (indexing) and **online** (querying).
- **Environment.** Windows 11, Python 3.12, CPU only (16 logical cores, 32 GB RAM).

**Visuals**
- `[VIDEO]` A screenshot of the home page of the application (sidebar + random gallery).

---

## 2. The dataset

**What to explain**

MIRFLICKR-25k is a set of 25,000 Flickr photos collected for research on image
retrieval (Huiskes & Lew, 2008). Before writing any code we inspected all the
files to learn their exact formats:

| Data | Location | Format found |
|---|---|---|
| Images | `data/mirflickr/im1.jpg … im25000.jpg` | JPEG, about 500 px on the longest side, 2.8 GB in total |
| Tags | `meta/tags/tagsN.txt` | one cleaned tag per line (lowercase, spaces removed, e.g. `governorsisland`) |
| EXIF | `meta/exif/exifN.txt` | lines alternate `-Field name` / value (camera, date, ISO, aperture…) |
| Licence | `meta/license/licenseN.txt` | `Key : value` lines (title, owner, licence) |
| Annotations | `data/annotations/*.txt` | one image id per line, one file per concept |
| MPEG-7 EHD | `data/mpeg7/edgehistogram/N.txt` | **150** floats, one per line, no header |
| MPEG-7 HTD | `data/mpeg7/homogeneoustexture/N.txt` | **43** floats, one per line, no header |

**Numbers**
- 25,000 images; **22,872** have at least one tag (2,128 have none); **223,535**
  (image, tag) pairs; **69,099** distinct tags.
- **24 concept labels** ("potential" labels, e.g. `dog.txt`), and for 14 of them a
  stricter "relevant" version (`dog_r1.txt`: the concept is salient). The
  dataset README announces 15 relevant labels, but `lake_r1.txt` is missing.
  **24,581** images have at least one label.
- The labels are very unbalanced:

`[TABLE]` Number of images per concept label (useful to explain the evaluation later):

| Label | Images | Label | Images | Label | Images |
|---|---|---|---|---|---|
| people | 10,373 | portrait | 3,931 | flower | 1,823 |
| structures | 9,992 | clouds | 3,700 | sea | 1,322 |
| plant_life | 8,763 | water | 3,331 | car | 1,177 |
| indoor | 8,313 | animals | 3,216 | food | 990 |
| sky | 7,912 | transport | 2,895 | river | 894 |
| female | 6,184 | night | 2,711 | lake | 791 |
| male | 6,081 | sunset | 2,135 | bird | 742 |
| tree | 4,683 | — | — | dog / baby | 684 / 259 |

- **Observation on HTD:** the standard MPEG-7 HTD has 62 values (mean, standard
  deviation, 30 energies, 30 energy deviations), but the provided files contain
  **43**. The layout is not documented, so we treat HTD as a 43-dimensional
  vector (see chapter 4).

**Visuals**
- A small grid of 6–8 sample images from the dataset (any images from the gallery in the video).

---

## 3. System architecture

**What to explain**

The system has two phases.

**Offline phase** (run once, `scripts/`):
1. `build_index.py` computes the Color, LBP and HOG descriptors of the 25,000
   images in parallel (8 processes), reads the provided EHD and HTD files,
   fills the SQLite database with paths, tags, EXIF, licences and labels, and
   builds one VP-Tree per (descriptor, metric) pair.
2. `extract_vgg16.py` computes the VGG16 deep features (in batches, with
   checkpoints so it can be interrupted and resumed).
3. Everything is saved in `artifacts/`:
   - `features_<descriptor>.npy` / `vgg16.npy`: one row per image (float32 matrices),
   - `vptree_<descriptor>_<metric>.joblib`: the 13 trees (only indices and bounds; the vectors are not duplicated),
   - `cbir.sqlite`: images, tags and labels tables,
   - `manifest.json`: descriptor parameters; `stats.json`: normalization values and build times.

**Online phase** (`backend/`): the FastAPI server loads all of these files once
at start-up (vectors in memory, trees, the VGG16 network for uploads). Each
query is then answered from memory and returned as JSON to the web client
(`frontend/`), which displays the results.

**Key design point: identical processing offline and online.** The descriptor
parameters are written to `manifest.json` during indexing and read back by the
server. An uploaded image therefore goes through exactly the same functions with
the same parameters. Proof: uploading `im1.jpg` (or `im123.jpg`) returns the same
image first at distance **0.0**.

**Database schema (SQLite)**

| Table | Columns | Role |
|---|---|---|
| `images` | id, filename, url, width, height, title, owner, license, camera, date_taken, exif_json | one row per image (path, not the image itself) |
| `tags` | image_id, tag (indexed) | keyword search |
| `labels` | image_id, label, kind (potential / relevant) | ground truth for the evaluation |

**Numbers**
- Offline build time: **7.8 min** in total (pixel descriptors 3.7 min on 8
  processes, metadata into SQLite 3.7 min, VP-Trees a few seconds).
- VGG16 extraction: **66 min** (6.1 images/s average on CPU; 11 images/s when
  the machine was not busy).
- Size of `artifacts/`: about **560 MB** (HOG alone: 378 MB, 25,000 × 3,780 floats).

**Visuals**
- The diagram of the README (section "How it works"), or a redrawn version.
- `[VIDEO]` The terminal at the start of the demo: `uvicorn backend.main:app --port 8000` and the line `Uvicorn running on http://127.0.0.1:8000`.

---

## 4. Visual descriptors

For each descriptor: what it captures, how it is computed, its size, and how it is compared.

`[TABLE]` Summary (put it at the start of the chapter):

| Descriptor | Source | Dimension | Captures | Default metric |
|---|---|---|---|---|
| Color (HSV) | computed | 128 | color distribution | histogram intersection |
| LBP | computed | 28 | micro-texture | L1 |
| HOG | computed | 3,780 | shape / edge layout | L2 |
| EHD (MPEG-7) | provided | 150 | edge directions per region | L1 |
| HTD (MPEG-7) | provided | 43 | texture (Gabor energies) | L1 |
| VGG16 | computed (CNN) | 512 | semantic content | L2 |

### 4.1 Color histogram (HSV)
- The image is converted from RGB to **HSV** (Hue, Saturation, Value). HSV
  separates the chromatic identity of a color (hue) from its purity and
  brightness, which matches human perception better than RGB.
- Joint 3-D histogram with **8 × 4 × 4 = 128 bins**: hue gets the most bins
  because it carries the color identity.
- The histogram is **L1-normalized** (it sums to 1), so it does not depend on the image size.
- Before computing, the image is resized so its longest side is 256 px (same for LBP).
- Strength: simple, fast, robust to rotation. Weakness: no spatial or semantic information (a red car and a red flower look alike).

### 4.2 LBP (Local Binary Patterns)
- For each pixel, P neighbours on a circle of radius R are compared with the
  centre pixel, giving a P-bit binary code that describes the local texture.
- We use the **uniform rotation-invariant** variant (riu2): codes with at most
  two 0/1 transitions are "uniform" and are labelled by their number of 1s;
  all other codes share one label. This gives **P + 2 bins** per scale and
  makes the descriptor invariant to rotation.
- Two scales: (P, R) = (8, 1) → 10 bins and (16, 2) → 18 bins, concatenated = **28 values**.
  P and R are configurable (`backend/config.py`).
- Each histogram is L1-normalized, then the vector is divided by the number of scales so it sums to 1.

### 4.3 HOG (Histogram of Oriented Gradients)
- The grey image is resized to **64 × 128** pixels (the classic detection window).
- Gradients are computed. Each **8 × 8** pixel cell accumulates a **9-bin**
  histogram of gradient orientations (0–180°), weighted by gradient magnitude.
- Cells are grouped into overlapping **2 × 2 blocks**, each normalized with
  **L2-Hys** (L2 normalization, clipping at 0.2, renormalization). This gives
  robustness to lighting changes.
- Size: 7 × 15 blocks × 36 values = **3,780 values**.
- Captures the global shape and edge layout of the image.

### 4.4 EHD (MPEG-7 Edge Histogram), provided
- The image is divided into 4 × 4 sub-images. Inside each, small blocks are
  classified into 5 edge types: vertical, horizontal, 45°, 135°, non-directional.
- 80 local bins + 5 global + 65 semi-global (rows, columns and groups of
  sub-images) = **150 values**. The global bins carry larger values, a standard weighting.
- Used as provided and compared with L1, the standard EHD matching measure.

### 4.5 HTD (MPEG-7 Homogeneous Texture), provided
- Based on the energy of Gabor filter responses at several scales and orientations.
- The files contain **43 values** instead of the standard 62, and the layout is
  not documented, so we treat the vector as an opaque 43-dimensional texture
  signature.
- Its dimensions have very different ranges, so each dimension is
  **z-score normalized** (mean 0, standard deviation 1, computed over the database).

### 4.6 VGG16 deep features
- VGG16 is a convolutional neural network (13 convolution layers in 5 blocks)
  trained on ImageNet (1.2 million images, 1,000 classes). We use the
  pretrained weights from torchvision and keep only the convolutional part.
- Pipeline: resize to **224 × 224** → ImageNet normalization → convolutional
  blocks → 512 feature maps of 7 × 7 → **global average pooling** → **512 values**
  → L2 normalization (unit length).
- On unit vectors, the L2 distance ranks images exactly like cosine similarity.
- Unlike hand-crafted descriptors, these features encode **semantic content**
  (objects, animals, scene types) learned from labelled data.

### 4.7 Uploaded images
- Color, LBP, HOG and VGG16 are computed on the fly for an uploaded image.
- EHD and HTD are provided files that exist only for database images, so the
  interface disables them for uploads (we chose not to approximate them).

**Visuals**
- `[VIDEO]` The descriptors table in the sidebar (checkboxes, metric, weight).
- `[VIDEO]` The same query image (e.g. the eagle, image **#11120**) with **Color only** and then **VGG16 only**. This is the most important visual of the report:
  Color returns orange/brown images (the background color), VGG16 returns birds.
- `[VIDEO]` Same comparison with the pink car (image **#21272**): Color returns pink images, VGG16 returns cars.
- Optional: an illustration of LBP (3×3 neighbourhood → binary code) and of HOG cells/blocks (draw or take from the course).

---

## 5. Similarity measures and descriptor combination

**What to explain**

**Distances implemented** (`backend/index/distances.py`):
- **L1** (Manhattan): `Σ |x_i − q_i|`
- **L2** (Euclidean): `√ Σ (x_i − q_i)²`
- **Histogram intersection**: `1 − Σ min(x_i, q_i)`, for normalized histograms.
  For histograms that sum to 1, it equals exactly `0.5 × L1`, so it is a true
  metric (it satisfies the triangle inequality) and can be used in the VP-Tree.
  This is why Color and LBP give the same results with L1 and intersection.

All three are **metrics**, which is required by the VP-Tree pruning rule (chapter 6).

**Combining several descriptors.** Each descriptor has its own scale: a Color
distance is around 0.8, an HTD L1 distance around 42. Adding them directly would
let one descriptor dominate. So each distance `d_i` is first divided by a scale
`s_i`, the **median distance between 2,000 random pairs** of database images,
computed offline. Then a weighted average is taken with the user's weights:

```
D(q, x) = ( Σ_i  w_i · d_i(q, x) / s_i ) / ( Σ_i w_i )
```

A weighted sum of metrics is still a metric.
- **Brute-force mode** computes D for all 25,000 images: the result is **exact**.
- **VP-Tree mode** asks each descriptor's tree for its max(10K, 200) nearest
  candidates, then re-ranks the union of candidates with the exact D. This is
  fast but **approximate**: an image that is not among the candidates of any
  single descriptor can be missed. The evaluation measures this: the top-20
  overlap with the exact result is between **71 % and 100 %** (chapter 9).

`[TABLE]` Normalization scales used (from `artifacts/stats.json`):

| Descriptor / metric | Scale s_i | Descriptor / metric | Scale s_i |
|---|---|---|---|
| color / intersection | 0.842 | ehd / L1 | 13.90 |
| lbp / L1 | 0.344 | htd / L1 | 42.16 |
| hog / L2 | 9.06 | vgg16 / L2 | 1.164 |

**Visuals**
- `[VIDEO]` A search with two descriptors ticked and different weights (e.g. Color w=1 + VGG16 w=2).
- `[VIDEO]` The tooltip when hovering a result, showing the distance of each descriptor.

---

## 6. Indexing: VP-Tree and brute force

**What to explain**

**Brute force** (`backend/index/bruteforce.py`): compute the distance from the
query to all N images and keep the K smallest. Cost: N distance computations
per query. It is vectorized with NumPy (processed in chunks of 2,048 rows to
limit memory) and serves as the reference for exactness and speed.

**VP-Tree** (Vantage-Point Tree, Yianilos 1993, `backend/index/vptree.py`). It
only uses distances, so it works with any metric.

*Construction* (recursive):
1. Choose a **vantage point** `vp` among the points of the node. We test 5 random
   candidates and keep the one whose distances to a sample are the most spread
   out, which gives more useful splits.
2. Compute `d(vp, x)` for every other point and split at the **median** `μ`:
   the *inner* child receives the points with `d ≤ μ`, the *outer* child those with `d ≥ μ`.
   The tree is therefore **balanced**.
3. Each child stores the range `[lo, hi]` of `d(vp, x)` over its points.
4. Stop when a node has at most **512 points**: it becomes a *leaf* (bucket),
   scanned with one vectorized distance computation.

*k-NN search*: keep the K best results found so far, with τ = the distance of
the K-th best (initially ∞).
- At an internal node, compute `dq = d(q, vp)`.
- **Triangle inequality**: every point x of a child with `d(vp, x) ∈ [lo, hi]`
  satisfies `d(q, x) ≥ max(lo − dq, dq − hi)`.
- If this lower bound is larger than τ, no point of that child can enter the
  top K, so the **whole subtree is skipped** (pruned).
- The nearer child is explored first, so τ shrinks quickly and more subtrees are pruned.
- The search is **exact**: it returns the same results as brute force. This was
  verified by `tests/test_vptree.py` (random data, L1/L2/intersection) and on
  100 % of the evaluation queries.

One tree is built per (descriptor, metric) pair: **13 trees**, built in 0.02 s
(LBP) to 1.7 s (HOG).

**Tuning the leaf size.** Our first trees used leaves of 32 points: they pruned
more, but the VP-Tree was slower than brute force, because each node visit runs
Python code while brute force is a single NumPy operation. Larger leaves mean
fewer Python steps and more vectorized work. We measured (N = 25,000, 30 queries):

`[TABLE]` Query time depending on the leaf size (% = distances computed):

| Descriptor | Brute force | Leaf 32 | Leaf 128 | Leaf 512 |
|---|---|---|---|---|
| Color | 7.3 ms | 16.0 ms (68 %) | 6.2 ms (73 %) | **4.5 ms** (79 %) |
| LBP | 0.8 ms | 2.6 ms (10 %) | 1.2 ms (14 %) | **0.7 ms** (21 %) |
| EHD | 10.2 ms | 12.2 ms (51 %) | 6.2 ms (57 %) | **3.9 ms** (62 %) |
| HTD | 1.1 ms | 6.5 ms (28 %) | 2.5 ms (34 %) | **1.3 ms** (39 %) |
| HOG | 195 ms | 280 ms (100 %) | — | 271 ms (100 %) |

→ We kept **512**: the fastest for every descriptor. This shows the trade-off
between "number of distance computations" and "overhead per computation".

**Visuals**
- A drawing of a VP-Tree: a vantage point, a circle of radius μ, inner/outer
  points, and a query circle of radius τ that does not intersect one side (= pruning). Draw it yourself or adapt one from the course.
- `[VIDEO]` The index selector in the sidebar (VP-Tree / Brute force).

---

## 7. Text search and fuzziness

**What to explain**

**Keyword search.** Keywords are compared with the Flickr tags stored in
SQLite (table `tags`, indexed on `tag`). Keywords are separated by commas and
normalized like the tags (lowercase, spaces removed).
- **OR** mode: the image has at least one of the keywords.
- **AND** mode: the image has all of them.
- Results are ranked by number of matched keywords.
- **Keywords + image**: the keywords first select the candidate images, which
  are then ranked by visual distance (brute force on this subset only).

**Typo tolerance (rapidfuzz).** Each keyword is compared with the 69,099
distinct tags using a similarity score from 0 to 100 (`fuzz.ratio`, based on
edit distance). The keyword is "expanded" to the tags above a cutoff (at most 10):

```
cutoff = 100 − fuzziness × 30      (fuzziness from 0 to 1)
```
- fuzziness 0 % → cutoff 100 → exact tags only,
- fuzziness 100 % → cutoff 70 → tolerant. Example: `sunst` matches `sunset`, `sunsets`, `sunstar`…
- The interface shows which tags each keyword actually matched.

**Fuzziness for visual search.** The same slider controls how strict the image
search is. Required rule: normalize the distances to [0, 1] over the whole
database, and keep a result only if its normalized distance is at most
`f × (normalized distance of the K-th result)`.

With min-max normalization `n(d) = (d − d_min) / (d_max − d_min)`, the
condition `n(d_i) ≤ f · n(d_K)` becomes:

```
(d_i − d_min) / (d_max − d_min)  ≤  f · (d_K − d_min) / (d_max − d_min)
⟺   (d_i − d_1) / (d_K − d_1)  ≤  f        (d_max cancels out; d_min = d_1, the best result)
```

So the rule can be applied **exactly without scanning the database for its
maximum**. It therefore also works with the VP-Tree.
- f = 100 % → the usual top-K,
- f = 50 % → only the results in the first half of the distance gap between the best and the K-th result,
- f = 0 % → only the best match (and exact ties).
- Example measured with VGG16, eagle #11120, K = 20: **20** results at 100 %, **8** at 50 %, **2** at 20 %.

**Visuals**
- `[VIDEO]` The keyword `sunset` (keyword-only search, the status bar shows "SQLite tag index").
- `[VIDEO]` The typo `sunst`: no result at 0 %, matched to `sunset` at 50 % (the query box shows `sunst → sunset`).
- `[VIDEO]` The same image query at fuzziness 100 %, 50 % and 20 % (the status bar shows "Results returned: 20 / 20", "8 / 20", "2 / 20").

---

## 8. The web application

**What to explain**

**Server (FastAPI, `backend/main.py`)**, JSON exchange:

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/config` | available descriptors, metrics, parameters |
| GET | `/api/random?n=` | random images for the gallery |
| GET | `/api/images/{id}` | metadata, tags and labels of one image |
| GET | `/api/tags/suggest?prefix=` | tag autocompletion |
| POST | `/api/search` | search by database image and/or keywords (JSON body) |
| POST | `/api/search/upload` | search with an uploaded image (file + JSON parameters) |

Images are served directly from their folder (`/images/im123.jpg`). Automatic
interactive documentation is available at `/docs`.

`[TABLE]` Example request / response (real values for query image #123, shortened):

```json
POST /api/search
{
  "image_id": 123,
  "descriptors": [{"name": "vgg16", "metric": "l2", "weight": 1}],
  "index": "vptree",
  "k": 20,
  "fuzziness": 100
}
```
```json
{
  "query": {"kind": "database", "image_id": 123, "url": "/images/im123.jpg"},
  "results": [
    {"rank": 1, "image_id": 19952, "url": "/images/im19952.jpg", "distance": 0.5157,
     "relative_distance": 0.0, "tags": ["..."]},
    {"rank": 2, "image_id": 15786, "url": "/images/im15786.jpg", "distance": 0.5577, "...": "..."}
  ],
  "index_used": "VP-Tree",
  "time_ms": 44.8,
  "distance_computations": 25000,
  "results_returned": 20
}
```

**Client (HTML/CSS/JavaScript, `frontend/`)**
- Sidebar: keywords (AND/OR), query image (id, random gallery, upload),
  descriptors table (checkbox, metric, weight), fuzziness slider, index type, K.
- Status bar: server search time, round-trip time, index used, number of distance
  computations, number of results returned.
- The query image is shown, then the results from most to least similar, with their distance.
- **Clicking a result starts a new search with it as the query.**
- Hovering a result shows the distance per descriptor and its tags.

**Visuals** (all `[VIDEO]`)
- The full interface after a search (status bar + query + results grid).
- An upload: the EHD/HTD rows greyed out, and the uploaded image found first at distance 0.000.
- Optional: the `/docs` page of FastAPI.

---

## 9. Evaluation: retrieval quality

**What to explain**

*Protocol* (`scripts/run_evaluation.py`)
- Ground truth: the 24 MIRFLICKR concept labels.
- **A result is relevant if it shares at least one concept label with the query.**
  The query itself is excluded from its results, and only images with at least one label are used as queries.
- **100 random queries** (fixed seed 42, so the results are reproducible).
- Metrics at K = 10 and K = 20:
  - **P@K** (precision) = relevant results in the top K / K,
  - **R@K** (recall) = relevant results in the top K / all relevant images in the database,
  - **AP@K** = (1 / min(K, R)) · Σ_{i≤K} P@i · rel_i, and **mAP@K** its mean over the queries.
- Brute-force search is used for the quality numbers (exact results).
- A **random-ranking baseline** is computed. Its precision is the average share
  of relevant images per query: **0.564**. This is high because the labels are
  very broad (*people* covers 10,373 images, *structures* 9,992…), so half the
  database is already "relevant" to a typical query. **All results must be read
  against 0.564.**

`[TABLE]` Results (100 queries, 25,000 images):

| Configuration | P@10 | P@20 | mAP@10 | mAP@20 |
|---|---|---|---|---|
| **VGG16 [L2]** | **0.892** | **0.888** | **0.855** | **0.841** |
| Color + LBP + VGG16 | 0.831 | 0.816 | 0.758 | 0.728 |
| Color + LBP + HOG | 0.727 | 0.712 | 0.623 | 0.591 |
| Color + LBP | 0.696 | 0.690 | 0.587 | 0.557 |
| LBP [L2] | 0.697 | 0.671 | 0.574 | 0.535 |
| LBP [L1] (= intersection) | 0.685 | 0.671 | 0.564 | 0.531 |
| All 5 hand-crafted | 0.683 | 0.676 | 0.573 | 0.539 |
| Color + HOG | 0.661 | 0.665 | 0.544 | 0.527 |
| Color [intersection] (= L1) | 0.650 | 0.646 | 0.517 | 0.499 |
| Color + EHD + HTD | 0.645 | 0.630 | 0.512 | 0.475 |
| Color [L2] | 0.642 | 0.646 | 0.514 | 0.496 |
| EHD [L2] | 0.578 | 0.577 | 0.426 | 0.399 |
| EHD [L1] | 0.573 | 0.571 | 0.422 | 0.393 |
| EHD + HTD | 0.560 | 0.567 | 0.401 | 0.380 |
| HOG [L1] | 0.559 | 0.561 | 0.417 | 0.400 |
| HTD [L2] | 0.544 | 0.552 | 0.396 | 0.372 |
| HTD [L1] | 0.528 | 0.542 | 0.377 | 0.358 |
| HOG [L2] | 0.508 | 0.508 | 0.369 | 0.345 |
| *Random baseline* | *0.564* | *0.564* | — | — |

**Analysis to write**
1. **VGG16 is far ahead** (P@10 = 0.892): the concept labels are *semantic*
   (dog, car, people…) and VGG16 was trained to recognize semantic content.
   Hand-crafted descriptors only see colors, textures and edges.
2. **Among hand-crafted descriptors, LBP and Color work best** (0.65–0.70),
   **EHD is slightly above chance**, and **HOG and HTD are at or below chance**.
   HOG describes a rigid global layout on a 64×128 window, which suits object
   detection but not general scenes.
3. **Combining helps hand-crafted descriptors** (Color + LBP + HOG = 0.727 > each
   alone) **but hurts VGG16** (Color + LBP + VGG16 = 0.831 < VGG16 alone =
   0.892): adding weaker descriptors dilutes the best one. Adding EHD + HTD to
   the hand-crafted ones also lowers the precision (0.727 → 0.683).
4. **L1 vs L2**: small differences. L1 is clearly better for HOG (0.559 vs 0.508).
   Intersection = L1 for normalized histograms (same results, as expected from theory).
5. **Recall is tiny** (R@20 ≈ 0.001) because each query has thousands of relevant
   images and we only retrieve 20. This is normal with such broad labels, so
   precision and mAP are the meaningful metrics here.
6. **Approximate combination in VP-Tree mode**: same quality as brute force
   (differences ≤ 0.01 in P@10), with a top-20 overlap of 71–100 % with the exact result.

`[TABLE]` Overlap between VP-Tree mode and exact brute force for combinations (top 20):

| Combination | Overlap |
|---|---|
| Color + HOG | 99.6 % |
| EHD + HTD | 96.3 % |
| Color + LBP | 94.4 % |
| Color + LBP + VGG16 | 92.0 % |
| Color + LBP + HOG | 87.5 % |
| Color + EHD + HTD | 80.9 % |
| All 5 hand-crafted | 70.6 % |

The more descriptors are combined, the more the per-descriptor candidates differ from the exact combined top-K.

**Visuals**
- `[FIGURE]` `report_assets/fig_retrieval_quality.png`: P@10 and P@20 for every configuration, with the random baseline.
- `[VIDEO]` The eagle (#11120) and pink car (#21272) comparisons as qualitative examples (Color vs VGG16).

---

## 10. Evaluation: speed, VP-Tree vs brute force

**What to explain**

Protocol: for each descriptor (default metric), build time and average query
time over the same 100 queries (k = 20), number of distance computations, and
exactness check. Then the same measurements for N = 1,000, 5,000, 10,000 and 25,000.

`[TABLE]` VP-Tree vs brute force (N = 25,000, k = 20):

| Descriptor (dim) | Build | VP-Tree | Brute force | Speed-up | Distances computed | Exact |
|---|---|---|---|---|---|---|
| LBP (28) | 0.02 s | 0.64 ms | 0.86 ms | ×1.35 | **21 %** | 100 % |
| HTD (43) | 0.06 s | 4.26 ms | 2.89 ms | ×0.68 | 38 % | 100 % |
| Color (128) | 0.06 s | 4.24 ms | 5.36 ms | ×1.26 | 82 % | 100 % |
| EHD (150) | 0.16 s | **13.1 ms** | 19.4 ms | **×1.48** | 61 % | 100 % |
| VGG16 (512) | 0.39 s | 34.7 ms | 33.4 ms | ×0.96 | 100 % | 100 % |
| HOG (3,780) | 1.73 s | 393 ms | 214 ms | ×0.54 | 100 % | 100 % |

**Analysis to write**
1. **The VP-Tree is always exact** (100 % of queries give the brute-force result).
2. **Pruning depends on the dimension.** With 28 dimensions (LBP), the tree
   computes only 21 % of the distances (and 18 % at N = 25k in the scalability
   test). With 512 (VGG16) or 3,780 (HOG) dimensions it computes 100 %: it prunes
   nothing. This is the **curse of dimensionality**: in high dimension all the
   distances between random points become similar, so the triangle-inequality
   bounds are never tight enough to exclude a subtree.
3. **Fewer computations do not always mean less time.** Our VP-Tree is written
   in Python and visits its nodes one by one, while brute force is one NumPy
   operation implemented in C. The VP-Tree wins in time when the pruning is
   strong *and* each distance is expensive enough (EHD ×1.48, LBP ×1.35,
   Color ×1.26), and loses when it cannot prune (HOG, VGG16).
4. **Scalability.** The share of distances computed by the VP-Tree **decreases
   when N grows** (LBP: 73 % at 1k → 18 % at 25k; HTD: 80 % → 36 %), while brute
   force always computes 100 %. The tree's advantage grows with the database size,
   which is what matters for "massive" databases.
5. Measured times vary by about ±30 % between runs depending on the computer's
   load; the ranking between methods stays the same.

`[TABLE]` Scalability: distances computed by the VP-Tree (% of the database):

| Descriptor | N = 1,000 | 5,000 | 10,000 | 25,000 |
|---|---|---|---|---|
| LBP | 73 % | 31 % | 24 % | 18 % |
| HTD | 80 % | 58 % | 46 % | 36 % |
| EHD | 94 % | 74 % | 70 % | 62 % |
| Color | 100 % | 92 % | 82 % | 85 % |
| HOG / VGG16 | 100 % | 100 % | 100 % | 100 % |

**Visuals**
- `[FIGURE]` `report_assets/fig_query_time.png`: query time per descriptor (log scale).
- `[FIGURE]` `report_assets/fig_distance_computations.png`: share of distances computed by the VP-Tree.
- `[FIGURE]` `report_assets/fig_scalability.png`: query time vs N for each descriptor.
- `[VIDEO]` LBP on the eagle #11120: switching VP-Tree ↔ Brute force gives the
  same results, but the status bar shows about **3,900 vs 25,000** distance computations.
- `[VIDEO]` VGG16 or HOG with the VP-Tree: 25,000 distance computations (no pruning).

---

## 11. Implementation challenges

Good material for a "difficulties and solutions" section:

| Problem | Cause | Solution |
|---|---|---|
| Undocumented data formats | No specification for the MPEG-7 files | Inspected all 50,000 files before coding: 150 values (EHD) and 43 values (HTD) everywhere, no header |
| HTD has 43 values instead of 62 | Non-standard export | Treated as an opaque vector, z-score normalized per dimension |
| Out-of-memory crash during extraction | 15 processes, each with its own OpenCV thread pool (16 threads) | 8 processes and one OpenCV thread per process |
| VP-Tree slower than brute force | Python overhead per node vs one vectorized NumPy call | Leaf size tuned from 32 to 512 (table in chapter 6) |
| Descriptors of different scales cannot be added | e.g. HTD distances ≈ 42, Color ≈ 0.8 | Divide by the median distance of random pairs before the weighted sum |
| Fuzziness needs the maximum distance over the database | Would require a full scan, defeating the index | Showed mathematically that the maximum cancels out (chapter 7) |
| VGG16 extraction takes about 1 hour on CPU | No GPU | Batches with prefetching, checkpoints every 320 images, resumable after an interruption (tested) |
| Uploads must be described like database images | Risk of different preprocessing | Parameters saved in `manifest.json` and reused by the server; checked with distance 0.0 on a re-uploaded image |

---

## 12. Limitations and future work

- **Python VP-Tree**: a compiled implementation (C++, Cython, Numba) would turn
  the reduction of distance computations into a real speed-up.
- **High dimension**: reduce HOG and VGG16 with PCA before indexing, or use
  approximate indexes (HNSW graphs, product quantization) that scale better
  than exact metric trees.
- **Evaluation**: the "shares at least one label" definition is broad (baseline
  0.564). The stricter `_r1` labels or graded relevance (number of shared labels)
  would give a finer evaluation.
- **Combination in VP-Tree mode** is approximate. More candidates per tree improve
  it at the cost of speed. An exact alternative would be one tree for the combined metric.
- **EHD/HTD for uploads**: they would require implementing the MPEG-7 extractors.
- **Other image bases**: the paths are configurable, so the "Objects" base of
  the course can be indexed with `--images-dir … --no-mpeg7`.
- **Packaging**: Docker / docker-compose was optional and not done.

---

## 13. Conclusion (key messages)

- A complete CBIR system was built **from scratch**: 6 descriptors, our own exact
  VP-Tree index, a JSON API, a web client, keyword search with typo tolerance,
  weighted combination, fuzziness, and an evaluation module.
- **Quality:** learned features (VGG16, P@10 = 0.892) clearly outperform
  hand-crafted ones (best 0.70) on semantic concepts. Among hand-crafted
  descriptors, texture (LBP) and color are the most useful.
- **Speed:** the VP-Tree is exact and prunes efficiently in low dimension, with an
  advantage that grows with the database size, but it suffers from the curse of
  dimensionality for HOG and VGG16.
- Choosing a descriptor is a trade-off between **what it captures**, its **size**,
  and **how well it can be indexed**.

---

## Checklist of visuals

### From the demo video (screenshots)
| # | Screenshot | Chapter |
|---|---|---|
| 1 | Terminal: `uvicorn backend.main:app --port 8000` + "Uvicorn running on…" | 3 |
| 2 | Home page: sidebar + random gallery | 1 / 8 |
| 3 | A basic search: status bar + query + results grid | 8 |
| 4 | Eagle #11120 with **Color** only | 4 / 9 |
| 5 | Eagle #11120 with **VGG16** only | 4 / 9 |
| 6 | Pink car #21272 with Color vs VGG16 (two screenshots) | 4 / 9 |
| 7 | Two descriptors combined with different weights | 5 |
| 8 | Tooltip with the per-descriptor distances | 5 |
| 9 | LBP: VP-Tree vs Brute force (status bar: ~3,900 vs 25,000 computations) | 10 |
| 10 | VGG16 or HOG with VP-Tree: 25,000 computations | 10 |
| 11 | Fuzziness 100 % / 50 % / 20 % ("Results returned" in the status bar) | 7 |
| 12 | Keyword search `sunset`, and the typo `sunst → sunset` | 7 |
| 13 | Upload: EHD/HTD greyed out + same image found at distance 0.000 | 4 / 8 |
| 14 | Click on a result → new search with it as the query | 8 |

### Ready-made figures (`report_assets/`)
| File | Chapter |
|---|---|
| `fig_retrieval_quality.png` | 9 |
| `fig_query_time.png` | 10 |
| `fig_distance_computations.png` | 10 |
| `fig_scalability.png` | 10 |

The raw numbers are in the CSV files next to them (`retrieval_metrics.csv`,
`benchmark_vptree_vs_bruteforce.csv`, `scalability.csv`) and summarized in
`evaluation_summary.md`.

### To draw (optional)
- A VP-Tree split with pruning (chapter 6).
- The LBP code computation on a 3×3 neighbourhood and the HOG cells/blocks (chapter 4).

---

## Appendix: commands, structure, glossary

**Commands (PowerShell, in the project folder)**
```powershell
.\.venv\Scripts\Activate.ps1
python -m tests.test_vptree            # VP-Tree = brute force (correctness test)
python -m scripts.build_index          # offline: descriptors, SQLite, VP-Trees (~8 min)
python -m scripts.extract_vgg16        # offline: VGG16 features (~1 h on CPU, resumable)
uvicorn backend.main:app --port 8000   # online: open http://127.0.0.1:8000
python -m scripts.run_evaluation       # evaluation: CSV + figures in report_assets/
```

**Project structure**
```
backend/
  config.py            paths and descriptor parameters
  main.py, schemas.py  FastAPI server and JSON models
  engine.py            online search: loading, combination, fuzziness
  descriptors/         color.py, lbp.py, hog.py, vgg16.py, mpeg7.py, preprocess.py
  index/               vptree.py, bruteforce.py, distances.py
  db/                  database.py (SQLite), metadata.py, fuzzy.py (typo tolerance)
frontend/              index.html, app.js, style.css
scripts/               build_index.py, extract_vgg16.py, run_evaluation.py
tests/                 test_vptree.py
report_assets/         CSV results and figures
```

**Glossary**
- **CBIR**: Content-Based Image Retrieval, searching images by their visual content.
- **Descriptor**: a vector of numbers that summarizes one aspect of an image (color, texture, shape, semantics).
- **k-NN**: k nearest neighbours, the K database vectors closest to the query vector.
- **Metric**: a distance that satisfies the triangle inequality `d(a, c) ≤ d(a, b) + d(b, c)`.
- **VP-Tree**: a tree that splits points by their distance to a vantage point and skips subtrees using the triangle inequality.
- **Brute force**: comparing the query with every image.
- **Pruning**: skipping a whole subtree because it cannot contain a better result.
- **Curse of dimensionality**: in high dimension, distances between points become similar, so indexes cannot prune.
- **P@K / R@K / mAP**: precision, recall and mean average precision on the first K results.
