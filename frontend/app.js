// Web client: builds the JSON query, calls the FastAPI server and renders the results.
"use strict";

const $ = (id) => document.getElementById(id);
const LABELS = {
  color: "Color (HSV)", lbp: "LBP", hog: "HOG", ehd: "EHD (MPEG-7)", htd: "HTD (MPEG-7)",
  vgg16: "VGG16 (deep features)",
};

const state = {
  config: null,
  query: null, // {kind: "database", imageId, url} | {kind: "upload", file, url}
};

// ------------------------------------------------------------------ setup
async function init() {
  state.config = await fetchJSON("/api/config");
  $("db-info").textContent = `${state.config.n_images.toLocaleString()} images indexed`;
  buildDescriptorTable();
  bindEvents();
  showRandom();
}

function buildDescriptorTable() {
  const tbody = document.querySelector("#descriptor-table tbody");
  for (const [name, info] of Object.entries(state.config.descriptors)) {
    const tr = document.createElement("tr");
    tr.dataset.name = name;
    tr.dataset.computable = info.computable;
    const metrics = info.metrics.map((m) => `<option value="${m}">${m}</option>`).join("");
    tr.innerHTML = `
      <td><input type="checkbox" class="d-on" ${name === "color" ? "checked" : ""}></td>
      <td title="${info.dim} dimensions">${LABELS[name] || name}</td>
      <td><select class="d-metric">${metrics}</select></td>
      <td><input type="number" class="d-weight" value="1" min="0" step="0.5"></td>`;
    tbody.appendChild(tr);
  }
}

function bindEvents() {
  $("search-form").addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
  $("btn-use-id").addEventListener("click", () => {
    const id = parseInt($("image-id").value, 10);
    if (id) setDatabaseQuery(id);
  });
  $("upload").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) setQuery({ kind: "upload", file, url: URL.createObjectURL(file) });
  });
  $("btn-clear-image").addEventListener("click", () => { setQuery(null); showRandom(); });
  $("fuzziness").addEventListener("input", updateFuzzHint);
  updateFuzzHint();
  let timer = null;
  $("keywords").addEventListener("input", (e) => {
    clearTimeout(timer);
    const last = e.target.value.split(",").pop().trim();
    if (last.length < 2) return;
    timer = setTimeout(async () => {
      const tags = await fetchJSON(`/api/tags/suggest?prefix=${encodeURIComponent(last)}`);
      const prefix = e.target.value.includes(",") ? e.target.value.replace(/[^,]*$/, " ") : "";
      $("tag-suggestions").innerHTML = tags
        .map((t) => `<option value="${prefix}${t.tag}">${t.count} images</option>`).join("");
    }, 200);
  });
}

// ------------------------------------------------------------------ query state
function setDatabaseQuery(imageId, search = false) {
  setQuery({ kind: "database", imageId, url: null });
  if (search) runSearch();
}

function setQuery(query) {
  state.query = query;
  const preview = $("query-preview");
  if (!query) {
    preview.className = "query-preview empty";
    preview.textContent = "No image selected";
    $("upload").value = "";
  } else {
    preview.className = "query-preview";
    // Database images get their URL from /api/images/{id} just below.
    preview.innerHTML = query.kind === "upload"
      ? `<img src="${query.url}" alt="uploaded query">`
      : `<img alt="query image ${query.imageId}">`;
    if (query.kind === "database") {
      $("image-id").value = query.imageId;
      fetchJSON(`/api/images/${query.imageId}`).then((info) => {
        query.url = info.url;
        preview.querySelector("img").src = info.url;
      }).catch(() => showError(`Unknown image id ${query.imageId}`));
    }
  }
  // Uploaded images only support descriptors computed from pixels.
  const upload = query && query.kind === "upload";
  $("upload-hint").hidden = !upload;
  for (const tr of document.querySelectorAll("#descriptor-table tbody tr")) {
    const blocked = upload && tr.dataset.computable !== "true";
    tr.classList.toggle("disabled", blocked);
    tr.querySelectorAll("input, select").forEach((el) => { el.disabled = blocked; });
    if (blocked) tr.querySelector(".d-on").checked = false;
  }
}

function updateFuzzHint() {
  const f = parseInt($("fuzziness").value, 10);
  $("fuzz-value").textContent = `${f}%`;
  const visual = f === 100
    ? "Image search: plain top-K."
    : f === 0
      ? "Image search: only the best match (and exact ties)."
      : `Image search: keeps results whose distance lies in the first ${f}% of the gap between the best match and the K-th match, so you may get fewer than K.`;
  const words = f === 0 ? "Keywords: exact tags only." : "Keywords: also matches tags with typos or variants (higher = more tolerant).";
  $("fuzz-hint").textContent = `${visual} ${words}`;
}

function collectParams() {
  const descriptors = [];
  for (const tr of document.querySelectorAll("#descriptor-table tbody tr")) {
    if (!tr.querySelector(".d-on").checked || tr.classList.contains("disabled")) continue;
    descriptors.push({
      name: tr.dataset.name,
      metric: tr.querySelector(".d-metric").value,
      weight: parseFloat(tr.querySelector(".d-weight").value) || 0,
    });
  }
  return {
    keywords: $("keywords").value.split(",").map((s) => s.trim()).filter(Boolean),
    keyword_mode: document.querySelector("input[name=kwmode]:checked").value,
    index: document.querySelector("input[name=index]:checked").value,
    k: parseInt($("k").value, 10),
    fuzziness: parseInt($("fuzziness").value, 10),
    descriptors,
  };
}

// ------------------------------------------------------------------ search
async function runSearch() {
  hideError();
  const params = collectParams();
  const q = state.query;
  if (!q && params.keywords.length === 0) return showError("Enter keywords or choose a query image.");
  if (q && params.descriptors.length === 0) return showError("Select at least one descriptor.");

  $("btn-search").disabled = true;
  const t0 = performance.now();
  try {
    let data;
    if (q && q.kind === "upload") {
      const form = new FormData();
      form.append("file", q.file);
      form.append("params", JSON.stringify(params));
      data = await fetchJSON("/api/search/upload", { method: "POST", body: form });
    } else {
      if (q) params.image_id = q.imageId;
      data = await fetchJSON("/api/search", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params),
      });
    }
    renderResults(data, performance.now() - t0);
  } catch (err) {
    showError(err.message);
  } finally {
    $("btn-search").disabled = false;
  }
}

function renderResults(data, roundTripMs) {
  const status = $("status");
  status.hidden = false;
  const descs = data.descriptors.map((d) => `${LABELS[d.name] || d.name} [${d.metric}, w=${d.weight}]`).join(" + ");
  status.innerHTML = `
    <span>Results returned: <b>${data.results_returned} / ${data.k_requested}</b> (fuzziness ${data.fuzziness}%)</span>
    <span>Search time: <b>${data.time_ms.toFixed(1)} ms</b></span>
    <span>Round trip: <b>${roundTripMs.toFixed(0)} ms</b></span>
    <span>Index: <b>${data.index_used}</b></span>
    ${data.distance_computations ? `<span>Distance computations: <b>${data.distance_computations.toLocaleString()}</b></span>` : ""}
    ${data.keyword_matches != null ? `<span>Keyword matches: <b>${data.keyword_matches.toLocaleString()}</b></span>` : ""}
    ${descs ? `<span>Descriptors: <b>${descs}</b></span>` : ""}`;

  const banner = $("query-banner");
  const q = data.query;
  const src = q.kind === "upload" ? state.query.url : q.url;
  banner.hidden = q.kind === "keywords" && !q.keywords.length;
  banner.innerHTML = (src ? `<img src="${src}" alt="query">` : "") +
    `<div><b>Query</b><br>${q.kind === "database" ? `database image #${q.image_id}` : q.kind === "upload" ? "uploaded image" : "keywords only"}` +
    (q.keywords.length ? `<br>keywords (${q.keyword_mode.toUpperCase()}): ${q.keywords.map((w) => {
      const tags = data.keyword_expansions[w] || [];
      return tags.length ? `${escapeHTML(w)} → ${tags.map(escapeHTML).join(", ")}` : `${escapeHTML(w)} → (no matching tag)`;
    }).join("; ")}` : "") + `</div>`;

  $("results-title").textContent = data.results.length
    ? `${data.results.length} result${data.results.length > 1 ? "s" : ""}, most similar first`
    : "No results";
  renderGrid(data.results);
}

function renderGrid(items) {
  const grid = $("grid");
  grid.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("div");
    card.className = "card";
    const score = item.distance != null ? `d = ${item.distance.toFixed(4)}`
      : item.matched_keywords != null ? `${item.matched_keywords} keyword(s)` : "";
    card.innerHTML = `<img loading="lazy" src="${item.url}" alt="image ${item.image_id}">
      <div class="meta"><span class="rank">${item.rank ? "#" + item.rank : "id " + item.image_id}</span><span>${score}</span></div>`;
    // Clicking a result relaunches a search with that image as the query.
    card.addEventListener("click", () => { setDatabaseQuery(item.image_id, true); window.scrollTo({ top: 0, behavior: "smooth" }); });
    card.addEventListener("mousemove", (e) => showTooltip(e, item));
    card.addEventListener("mouseleave", () => { $("tooltip").hidden = true; });
    grid.appendChild(card);
  }
}

async function showRandom() {
  $("status").hidden = true;
  $("query-banner").hidden = true;
  $("results-title").textContent = "Pick a query image (click any image)";
  const items = await fetchJSON("/api/random?n=30");
  renderGrid(items);
}

// ------------------------------------------------------------------ utils
function showTooltip(e, item) {
  const tip = $("tooltip");
  const parts = [`<b>Image #${item.image_id}</b>`];
  if (item.relative_distance != null) parts.push(`relative distance: ${item.relative_distance.toFixed(2)} (0 = best, 1 = K-th)`);
  if (item.distances) {
    for (const [k, v] of Object.entries(item.distances)) parts.push(`${k}: ${v.toFixed(4)}`);
  }
  if (item.tags && item.tags.length) parts.push(`tags: ${item.tags.slice(0, 12).map(escapeHTML).join(", ")}`);
  tip.innerHTML = parts.join("<br>");
  tip.hidden = false;
  tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 300) + "px";
  tip.style.top = e.clientY + 14 + "px";
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const body = await res.json(); msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function showError(msg) { const el = $("error"); el.textContent = msg; el.hidden = false; }
function hideError() { $("error").hidden = true; }

init();
