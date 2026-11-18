import { parseNeighbors, recommendForNewUser, similarMovies, splitTitle } from "./engine.js";

const $ = (sel, root = document) => root.querySelector(sel);
const app = $("#app");
const D = { cat: null, users: null, metrics: null, nb: null, nbPromise: null };

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n).toLocaleString("en-US");
const stars = (v) => `<span class="stars" style="--p:${Math.max(0, Math.min(5, v)) * 20}%" aria-hidden="true">★★★★★</span>`;

// ------------------------------------------------------------------ data loading
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url} (${r.status})`);
  return r.json();
}
async function loadCore() {
  [D.cat, D.users, D.metrics] = await Promise.all([
    getJSON("/data/catalog.json"),
    getJSON("/data/users.json"),
    getJSON("/data/metrics.json").catch(() => null),
  ]);
  D.cat.search = D.cat.title.map((t) => t.toLowerCase());
  D.cat.byPopularity = D.cat.id.map((_, i) => i).sort((a, b) => D.cat.nt[b] - D.cat.nt[a]);
}
function loadNeighbors() {
  if (!D.nbPromise) {
    D.nbPromise = fetch("/data/neighbors.bin")
      .then((r) => { if (!r.ok) throw new Error("Could not load similarity data"); return r.arrayBuffer(); })
      .then((buf) => (D.nb = parseNeighbors(buf, D.cat.id.length, D.cat.k)))
      .catch((e) => { D.nbPromise = null; throw e; });
  }
  return D.nbPromise;
}

// ------------------------------------------------------------------ movie rendering
function hue(text) {
  let h = 0;
  for (const c of String(text)) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return h % 360;
}
const SCORE_KINDS = {
  rating: ["Predicted", (v) => `★ ${v.toFixed(2)}`, (v) => v / 5],
  yours: ["You rated", (v) => `★ ${v.toFixed(1)}`, (v) => v / 5],
  similarity: ["Match", (v) => `${(v * 100).toFixed(0)}%`, (v) => v],
};
/** rows: [[position, score], ...]. mode: rating | yours | similarity. */
function cards(rows, { mode = "rating", numbered = true, compact = false } = {}) {
  if (!rows.length) return `<div class="empty">No results to show for this request.</div>`;
  const [label, fmtScore, frac] = SCORE_KINDS[mode];
  const body = rows.map(([p, score], i) => {
    const { title, year } = splitTitle(D.cat.title[p]);
    const h = hue(D.cat.title[p]);
    const genres = D.cat.genres[p].split("|").filter((g) => g && g !== "(no genres listed)").slice(0, 3);
    const width = Math.max(0, Math.min(1, frac(score))) * 100;
    const avg = D.cat.avg[p] ? D.cat.avg[p].toFixed(2) : "n/a";
    return `<div class="mcard">
      <div class="poster" style="background:linear-gradient(145deg,hsl(${h},58%,38%),hsl(${(h + 40) % 360},55%,16%))">${numbered ? `<span class="rank">${i + 1}</span>` : ""}<span class="year">${esc(year)}</span></div>
      <div class="mbody"><div class="mtitle">${esc(title)}</div>
        <div class="chips">${genres.map((g) => `<span class="chip" style="--h:${hue(g)}">${esc(g)}</span>`).join("")}</div>
        <div class="score"><span class="lbl">${label}</span><span class="val">${fmtScore(score)}</span></div>
        <div class="bar"><i style="width:${width.toFixed(0)}%"></i></div>
        <div class="sub">Avg rating ${avg} &middot; ${fmt(D.cat.n[p])} ratings</div></div>
    </div>`;
  }).join("");
  return `<div class="grid${compact ? " compact" : ""}">${body}</div>`;
}
function csvButton(id, rows, scoreName) {
  return `<button class="btn ghost" id="${id}" type="button">Download as CSV</button>`;
}
function wireCsv(id, rows, scoreName, filename) {
  const btn = document.getElementById(id);
  if (!btn) return;
  btn.onclick = () => {
    const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
    const lines = ["movieId,title,genres,avg_rating,n_ratings," + scoreName];
    for (const [p, s] of rows) lines.push([D.cat.id[p], q(D.cat.title[p]), q(D.cat.genres[p]), D.cat.avg[p], D.cat.n[p], s].join(","));
    const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: filename + ".csv" });
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
}
const kpis = (items) => `<div class="kpis">${items.map(([l, v, d]) => `<div class="kpi"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div><div class="hint">${esc(d)}</div></div>`).join("")}</div>`;

// ------------------------------------------------------------------ movie search combobox
function movieSearch(host, { placeholder, minRatings = 50, onPick, clear = false, id }) {
  host.classList.add("search");
  host.innerHTML = `<input id="${id}" type="text" role="combobox" aria-expanded="false" aria-autocomplete="list" aria-controls="${id}-list" autocomplete="off" spellcheck="false" placeholder="${esc(placeholder)}" aria-label="${esc(placeholder)}"><ul class="suggest" id="${id}-list" role="listbox" hidden></ul>`;
  const input = $("input", host), list = $("ul", host);
  let items = [], active = -1;
  const close = () => { list.hidden = true; input.setAttribute("aria-expanded", "false"); active = -1; };
  const paint = () => {
    list.innerHTML = items.length
      ? items.map((p, i) => { const t = splitTitle(D.cat.title[p]); return `<li role="option" id="${id}-o${i}" data-i="${i}" aria-selected="${i === active}"><span>${esc(t.title)} <small>${t.year}</small></span><small>${fmt(D.cat.nt[p])} ratings</small></li>`; }).join("")
      : `<li class="none">No movie matches that. Try fewer letters.</li>`;
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
  };
  const search = () => {
    const tokens = input.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    items = [];
    for (const p of D.cat.byPopularity) {
      if (D.cat.nt[p] < minRatings) break;
      if (tokens.every((t) => D.cat.search[p].includes(t))) { items.push(p); if (items.length === 8) break; }
    }
    active = -1;
    paint();
  };
  const pick = (p) => {
    close();
    input.value = clear ? "" : splitTitle(D.cat.title[p]).title;
    onPick(p);
  };
  input.addEventListener("focus", search);
  input.addEventListener("input", search);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (list.hidden) search();
      active = (active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % Math.max(items.length, 1);
      paint();
      input.setAttribute("aria-activedescendant", `${id}-o${active}`);
      $(`#${id}-o${active}`, list)?.scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter" && active >= 0) { e.preventDefault(); pick(items[active]); }
    else if (e.key === "Escape") close();
  });
  list.addEventListener("mousedown", (e) => { const li = e.target.closest("li[data-i]"); if (li) { e.preventDefault(); pick(items[+li.dataset.i]); } });
  input.addEventListener("blur", close);
  return { input, setValue: (p) => { input.value = splitTitle(D.cat.title[p]).title; } };
}

// ------------------------------------------------------------------ pages
function pageHome() {
  const m = D.metrics?.dataset;
  const nUsers = Object.keys(D.users).length;
  app.innerHTML = `
    <div class="hero">
      <h1>Find your next favourite movie</h1>
      <p>Personalised recommendations built from the rating behaviour of hundreds of viewers, using user-based and item-based collaborative filtering.</p>
      ${["MovieLens ml-latest-small", "User-based CF", "Item-based CF", "Cold-start support", "RMSE / MAE evaluated"].map((x) => `<span class="pill">${x}</span>`).join("")}
    </div>
    ${kpis([
      ["Users", fmt(nUsers), "in the training data"],
      ["Movies", fmt(D.cat.id.length), "in the rating matrix"],
      ["Training ratings", m ? fmt(m.n_train) : "n/a", "80% of all ratings"],
      ["Matrix sparsity", m ? (m.sparsity * 100).toFixed(1) + "%" : "n/a", "cells with no rating"],
    ])}
    <h2>How it works</h2>
    <div class="steps">
      <div class="step"><div class="n">1</div><b>Learn from ratings</b><span>Each person's ratings are compared with everyone else's to find similar tastes (or similar movies).</span></div>
      <div class="step"><div class="n">2</div><b>Predict what you would rate</b><span>Ratings from your closest neighbours are averaged, weighted by how similar they are to you.</span></div>
      <div class="step"><div class="n">3</div><b>Rank unseen movies</b><span>Movies you already rated are removed and the highest predicted ones become your top picks.</span></div>
    </div>
    <div class="quick">
      <h2>Try it: name a movie you love</h2>
      <div id="hero-search"></div>
      <div class="chips" id="hero-chips" aria-label="Try one of these"></div>
      <div id="hero-out" aria-live="polite"></div>
    </div>
    <h2>Start here</h2>
    <div class="cta-row">
      <a class="btn" href="#/recommend">🎯  Recommendations for a user</a>
      <a class="btn ghost" href="#/similar">🎞️  Find similar movies</a>
      <a class="btn ghost" href="#/new">🆕  I'm a new user</a>
    </div>`;
  const out = $("#hero-out");
  const show = async (p) => {
    out.innerHTML = `<p class="muted">Finding movies like this…</p>`;
    try {
      await loadNeighbors();
    } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}. Check your connection and try again.</div>`; return; }
    const rows = similarMovies(D.cat, D.nb, p, 6);
    out.innerHTML = `<h3>Because you like ${esc(splitTitle(D.cat.title[p]).title)}</h3>${cards(rows, { mode: "similarity" })}
      <p><a href="#/similar">See more similar movies</a></p>`;
    out.firstElementChild.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  };
  movieSearch($("#hero-search"), { id: "hs", placeholder: "Type a movie, for example Toy Story", onPick: show });
  $("#hero-chips").innerHTML = D.cat.byPopularity.slice(0, 5).map((p) => `<button class="chip" style="--h:350" type="button" data-p="${p}">${esc(splitTitle(D.cat.title[p]).title)}</button>`).join("");
  $("#hero-chips").onclick = (e) => { const b = e.target.closest("[data-p]"); if (b) { $("#hs").value = b.textContent; show(+b.dataset.p); } };
  loadNeighbors().catch(() => {});
}

function pageRecommend() {
  const ids = Object.keys(D.users).map(Number).sort((a, b) => a - b);
  const st = pageRecommend.state ||= { user: ids[0], n: 10, method: "User-based" };
  app.innerHTML = `
    <h1>Recommendations</h1>
    <p class="sub">Pick a viewer from the dataset to see what they should watch next.</p>
    <div class="controls">
      <label class="field">Viewer (userId)<select id="u">${ids.map((u) => `<option ${u === st.user ? "selected" : ""}>${u}</option>`).join("")}</select></label>
      <label class="field">Number of movies: <span id="nv">${st.n}</span><input id="n" type="range" min="5" max="20" value="${st.n}"></label>
      <div class="field">Method<div class="seg" id="m" role="group" aria-label="Method">${["User-based", "Item-based", "Side by side"].map((x) => `<button type="button" aria-pressed="${x === st.method}">${x}</button>`).join("")}</div></div>
    </div>
    <div id="out"></div>`;
  const draw = () => {
    const u = D.users[st.user], n = st.n;
    const ur = u.user.slice(0, n), ir = u.item.slice(0, n);
    let html = kpis([
      ["Movies rated", fmt(u.n), "in the training data"],
      ["Average rating given", u.mean.toFixed(2), "out of 5"],
      ["Favourite genres", u.genres.join(", ") || "n/a", "by number of ratings"],
    ]);
    if (st.method === "User-based") {
      html += `<h2>Top ${ur.length} for viewer ${st.user}, from viewers with similar taste</h2>${cards(ur)}<p>${csvButton("csv1")}</p>`;
    } else if (st.method === "Item-based") {
      html += `<h2>Top ${ir.length} for viewer ${st.user}, similar to movies they rated</h2>${cards(ir)}<p>${csvButton("csv1")}</p>`;
    } else {
      const both = ur.filter(([p]) => ir.some(([q]) => q === p)).length;
      html += `<div class="callout">${both} of ${n} movies appear in both lists.</div>
        <div class="two"><div><h3>User-based</h3>${cards(ur, { compact: true })}</div><div><h3>Item-based</h3>${cards(ir, { compact: true })}</div></div>`;
    }
    html += `<details><summary>Why these? Taste neighbours and this viewer's favourites</summary><div class="two">
      <div><h3>Most similar viewers</h3><div class="tablewrap"><table><thead><tr><th>userId</th><th>similarity</th></tr></thead><tbody>${u.peers.map(([id, s]) => `<tr><td>${id}</td><td>${s.toFixed(3)}</td></tr>`).join("")}</tbody></table></div></div>
      <div><h3>Highest rated by this viewer</h3>${cards(u.hist, { mode: "yours", numbered: false, compact: true })}</div></div></details>`;
    $("#out").innerHTML = html;
    if (st.method !== "Side by side") wireCsv("csv1", st.method === "User-based" ? ur : ir, "predicted_rating", `viewer${st.user}_${st.method.toLowerCase()}`);
  };
  $("#u").onchange = (e) => { st.user = +e.target.value; draw(); };
  $("#n").oninput = (e) => { st.n = +e.target.value; $("#nv").textContent = st.n; draw(); };
  $("#m").onclick = (e) => {
    const b = e.target.closest("button"); if (!b) return;
    st.method = b.textContent; [...$("#m").children].forEach((x) => x.setAttribute("aria-pressed", x === b)); draw();
  };
  draw();
}

function pageSimilar() {
  const st = pageSimilar.state ||= { movie: D.cat.byPopularity[0], n: 10 };
  app.innerHTML = `
    <h1>Similar movies</h1>
    <p class="sub">Similarity comes from who rated the movies alike, not from genre labels.</p>
    <div class="controls">
      <label class="field" style="flex:1;min-width:260px">Pick or type a movie<span id="sm"></span></label>
      <label class="field">Results: <span id="nv">${st.n}</span><input id="n" type="range" min="5" max="20" value="${st.n}"></label>
    </div>
    <div id="out" aria-live="polite"></div>`;
  const box = movieSearch($("#sm"), { id: "sim", placeholder: "Search movies", onPick: (p) => { st.movie = p; draw(); } });
  box.setValue(st.movie);
  $("#n").oninput = (e) => { st.n = +e.target.value; $("#nv").textContent = st.n; draw(); };
  const draw = async () => {
    const out = $("#out");
    out.innerHTML = `<p class="muted">Finding similar movies…</p>`;
    try { await loadNeighbors(); } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}. Check your connection and try again.</div>`; return; }
    const rows = similarMovies(D.cat, D.nb, st.movie, st.n);
    if (!rows.length) { out.innerHTML = `<div class="warn">This movie has no similar movies with enough shared raters.</div>`; return; }
    out.innerHTML = `<h2>If you liked ${esc(splitTitle(D.cat.title[st.movie]).title)}</h2>${cards(rows, { mode: "similarity" })}<p>${csvButton("csv1")}</p>`;
    wireCsv("csv1", rows, "similarity", `similar_to_${D.cat.id[st.movie]}`);
  };
  draw();
}

function pageNew() {
  const st = pageNew.state ||= { ratings: new Map() };
  const max = D.cat.coldRecommended;
  app.innerHTML = `
    <h1>New here? Rate a few movies</h1>
    <p class="sub">Choose up to ${max} movies you know and rate them. At least ${D.cat.coldMin} are required. Nothing is saved and no account is created.</p>
    <div class="controls"><label class="field" style="flex:1;min-width:260px">Search and add a movie<span id="ns"></span></label></div>
    <div id="rates"></div>
    <p><button class="btn" id="go" type="button">Get my recommendations</button></p>
    <div id="out" aria-live="polite"></div>`;
  const rates = $("#rates"), go = $("#go");
  const drawRates = () => {
    go.disabled = st.ratings.size === 0;
    rates.innerHTML = st.ratings.size ? `<h3>Your ratings</h3><div class="rate-grid">${[...st.ratings].map(([p, r]) => `
      <div class="rate"><div class="top"><span>${esc(splitTitle(D.cat.title[p]).title)}</span><button class="x" type="button" data-rm="${p}" aria-label="Remove ${esc(splitTitle(D.cat.title[p]).title)}">×</button></div>
        <input type="range" min="0.5" max="5" step="0.5" value="${r}" data-p="${p}" aria-label="Rating for ${esc(splitTitle(D.cat.title[p]).title)}">
        <div>${stars(r)} <output>${r.toFixed(1)}</output></div></div>`).join("")}</div>` : "";
  };
  rates.addEventListener("input", (e) => {
    const p = e.target.dataset.p; if (p === undefined) return;
    const v = +e.target.value; st.ratings.set(+p, v);
    const box = e.target.closest(".rate");
    $("output", box).textContent = v.toFixed(1); $(".stars", box).style.setProperty("--p", v * 20 + "%");
  });
  rates.addEventListener("click", (e) => { const b = e.target.closest("[data-rm]"); if (b) { st.ratings.delete(+b.dataset.rm); drawRates(); } });
  const search = movieSearch($("#ns"), { id: "ns-in", placeholder: "Search and add a movie", minRatings: 50, clear: true,
    onPick: (p) => {
      if (st.ratings.size >= max && !st.ratings.has(p)) { $("#out").innerHTML = `<div class="warn">You can rate up to ${max} movies. Remove one to add another.</div>`; return; }
      $("#out").innerHTML = ""; st.ratings.set(p, st.ratings.get(p) ?? 4); drawRates();
    } });
  go.onclick = async () => {
    const out = $("#out");
    try { await loadNeighbors(); } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}. Check your connection and try again.</div>`; return; }
    try {
      const res = recommendForNewUser(D.cat, D.nb, st.ratings, 10);
      out.innerHTML = res.warnings.map((w) => `<div class="warn">${esc(w)}</div>`).join("") +
        (res.usedFallback ? `<div class="callout">Showing popular movies because a personalised list was not possible.</div>` : "") +
        `<h2>Picked for you</h2>${cards(res.recs)}<p>${csvButton("csv1")}</p>`;
      wireCsv("csv1", res.recs, "predicted_rating", "new_user_recommendations");
    } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
  };
  drawRates();
  loadNeighbors().catch(() => {});
}

function pagePerformance() {
  const M = D.metrics;
  if (!M) { app.innerHTML = `<h1>Model performance</h1><div class="warn">No evaluation results found. Run <code>python scripts/train.py</code> and then <code>python scripts/export_static.py</code>.</div>`; return; }
  const { config: c, dataset: ds } = M;
  const all = M.results;
  const common = all.filter((r) => r.subset === "common records");
  const best = common.reduce((a, b) => (b.rmse < a.rmse ? b : a));
  const cfOnly = common.filter((r) => !r.model.includes("Baseline"));
  const bestCf = cfOnly.reduce((a, b) => (b.rmse < a.rmse ? b : a));
  const cov = (model) => all.find((r) => r.model === model && r.subset === "all predictable").coverage;
  const bars = (key) => {
    const top = Math.max(...common.map((r) => r[key])) * 1.1;
    const bestKey = common.reduce((a, b) => (b[key] < a[key] ? b : a)).model;
    return `<div class="hbars">${common.map((r) => `<div class="hbar${r.model === bestKey ? " best" : ""}"><span>${esc(r.model)}</span><div class="track"><div class="fill" style="width:${(r[key] / top) * 100}%"></div></div><span class="val">${r[key].toFixed(3)}</span></div>`).join("")}</div>`;
  };
  const cols = ["model", "subset", "rmse", "mae", "coverage", "n_predicted", "n_test"];
  app.innerHTML = `
    <h1>Model performance</h1>
    <p class="sub">${esc(c.split_strategy.replace("_", "-"))} split, ${fmt(ds.n_train)} training and ${fmt(ds.n_test)} test ratings, seed ${c.seed}. Models are fitted on training data only.</p>
    ${kpis(common.map((r) => [r.model, r.rmse.toFixed(3), `RMSE, MAE ${r.mae.toFixed(3)}, coverage ${(cov(r.model) * 100).toFixed(1)}%`]))}
    <div class="callout">${best.model.includes("Baseline")
      ? `The item-mean baseline has the lowest RMSE here. The best collaborative model is ${esc(bestCf.model)} (${bestCf.rmse.toFixed(3)}). Baselines are strong on sparse data, so this comparison matters.`
      : `Lowest RMSE: ${esc(best.model)} (${best.rmse.toFixed(3)}), measured on the same ${fmt(best.n_predicted)} test ratings for every model.`}</div>
    <div class="two"><div><h3>RMSE (lower is better)</h3>${bars("rmse")}</div><div><h3>MAE (lower is better)</h3>${bars("mae")}</div></div>
    <details><summary>Full results table</summary><div class="tablewrap"><table><thead><tr>${cols.map((k) => `<th>${k}</th>`).join("")}</tr></thead><tbody>${all.map((r) => `<tr>${cols.map((k) => `<td>${typeof r[k] === "number" && !Number.isInteger(r[k]) ? r[k].toFixed(4) : esc(r[k])}</td>`).join("")}</tr>`).join("")}</tbody></table></div></details>
    <details><summary>How to read these metrics</summary><div class="prose"><ul>
      <li><b>RMSE</b> = sqrt(mean((actual − predicted)²)). Penalises large mistakes more.</li>
      <li><b>MAE</b> = mean(|actual − predicted|). The average miss in stars.</li>
      <li><b>Coverage</b> = share of test ratings the model could predict. A model that abstains often can look more accurate.</li>
      <li><b>Common records</b> = only test ratings every model predicted, which is the fair head-to-head.</li></ul></div></details>`;
}

function pageAbout() {
  app.innerHTML = `
    <div class="about-logo"><img src="/assets/logo.svg" width="72" height="72" alt=""><h1>About this project</h1></div>
    <div class="prose">
      <p><b>Intelligent Movie Recommendation System</b> is a collaborative-filtering recommender built on the MovieLens <i>ml-latest-small</i> dataset.</p>
      <h3>Pipeline</h3>
      <ol>
        <li>Clean ratings (types, ranges, duplicates, unknown movies).</li>
        <li>Split 80/20 per user, <i>before</i> any model sees the data.</li>
        <li>Build the user-item matrix (missing = unrated, never zero).</li>
        <li>Fit user-based and item-based models on the training part only.</li>
        <li>Evaluate on the held-out ratings with RMSE, MAE and coverage.</li>
        <li>Export the trained model as static files so the website runs on Vercel with no server.</li>
      </ol>
      <p><b>Method:</b> cosine similarity on mean-centred ratings, shrunk by overlap size. Prediction = mean + similarity-weighted deviations from the top-K neighbours.</p>
      <p><b>New users:</b> the ratings you enter are scored in your browser with the item-item similarities. Nothing is sent anywhere.</p>
      <p><b>Known limitations:</b> no genre or content features, no time awareness, popularity bias, and new users need a few ratings. Offline RMSE does not guarantee viewer satisfaction.</p>
      <p class="muted">Stack: Python, pandas, NumPy, scikit-learn for training. HTML, CSS and JavaScript for the website, hosted on Vercel.</p>
    </div>`;
}

// ------------------------------------------------------------------ router
const routes = { "": ["home", pageHome], recommend: ["recommend", pageRecommend], similar: ["similar", pageSimilar], new: ["new", pageNew], performance: ["performance", pagePerformance], about: ["about", pageAbout] };
function render() {
  const key = location.hash.replace(/^#\/?/, "");
  const [name, page] = routes[key] || routes[""];
  document.querySelectorAll("#nav a").forEach((a) => (a.dataset.route === name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  page();
  window.scrollTo(0, 0);
}
async function start() {
  try {
    await loadCore();
  } catch (e) {
    app.innerHTML = `<div class="err">${esc(e.message)}. Reload the page to try again.</div>`;
    return;
  }
  window.addEventListener("hashchange", render);
  render();
}
start();
