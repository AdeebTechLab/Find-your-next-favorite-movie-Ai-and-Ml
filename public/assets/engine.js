// Browser port of src/item_based_cf.py (_raw_score + _rank) and src/cold_start.py.
// Pure functions: no DOM, so it can be tested in Node against the Python model.

export const MIN_RATING = 0.5;
export const MAX_RATING = 5.0;
export const RATING_STEP = 0.5;

/** Neighbour lists from neighbors.bin: uint16 indices followed by uint16 similarities (x65535). */
export function parseNeighbors(buffer, n, k) {
  return {
    k,
    idx: new Uint16Array(buffer, 0, n * k),
    sim: new Uint16Array(buffer, n * k * 2, n * k),
  };
}

/** Same maths as ItemBasedCF._raw_score. `ratings` is a Map(position -> rating). */
export function rawScores(cat, nb, ratings) {
  const n = cat.id.length, k = nb.k;
  const out = new Float64Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    let num = 0, den = 0, support = 0;
    const base = i * k;
    for (let j = 0; j < k; j++) {
      const nbr = nb.idx[base + j];
      const r = ratings.get(nbr);
      if (r === undefined) continue;
      const w = nb.sim[base + j] / 65535;
      if (w > 0) support++;
      num += w * (r - cat.mu[nbr]);
      den += w;
    }
    if (support < cat.minNeighbors || den <= 1e-9) continue;
    out[i] = cat.mu[i] + num / (den + cat.damping);
  }
  return out;
}

/** Top-n positions by score, best first, ties broken by position (stable), like top_n_positions. */
export function topN(scores, n) {
  const valid = [];
  for (let i = 0; i < scores.length; i++) if (!Number.isNaN(scores[i])) valid.push(i);
  valid.sort((a, b) => scores[b] - scores[a] || a - b);
  return valid.slice(0, n);
}

export const clip = (x) => Math.min(MAX_RATING, Math.max(MIN_RATING, x));
export const round2 = (x) => Math.round(x * 100) / 100;

/** Check {position: rating} pairs the way validate_new_user_ratings does. Throws Error with a readable message. */
export function validateRatings(cat, ratings) {
  const warnings = [];
  const clean = new Map();
  for (const [pos, rating] of ratings) {
    const scaled = rating / RATING_STEP;
    if (rating < MIN_RATING || rating > MAX_RATING || Math.abs(scaled - Math.round(scaled)) > 1e-9) {
      warnings.push(`Ignored “${cat.title[pos]}”: rating ${rating} must be ${MIN_RATING}-${MAX_RATING} in steps of ${RATING_STEP}.`);
      continue;
    }
    clean.set(pos, rating);
  }
  if (clean.size === 0) throw new Error("None of the ratings could be used. " + warnings.join(" "));
  if (clean.size < cat.coldMin) {
    throw new Error(`Rate at least ${cat.coldMin} movies. You have ${clean.size} usable ${clean.size === 1 ? "rating" : "ratings"}. ${warnings.join(" ")}`.trim());
  }
  if (clean.size < cat.coldRecommended) warnings.push(`Only ${clean.size} ratings: the picks will be less personal than with ${cat.coldRecommended}.`);
  if (new Set(clean.values()).size === 1) warnings.push("All your ratings are the same, which says little about what you dislike.");
  const thin = [...clean.keys()].filter((p) => cat.nt[p] < cat.minRatings).length;
  if (thin) warnings.push(`${thin} chosen movie${thin > 1 ? "s have" : " has"} very few ratings in the data, so ${thin > 1 ? "they carry" : "it carries"} little signal.`);
  return { clean, warnings };
}

/** Non-personalised fallback: damped average rating (popular_movies). */
export function popular(cat, exclude, n) {
  const s = new Float64Array(cat.id.length);
  for (let i = 0; i < s.length; i++) s[i] = cat.nt[i] < cat.minRatings || exclude.has(i) ? NaN : cat.mu[i];
  return topN(s, n).map((p) => [p, round2(clip(s[p]))]);
}

/** Cold-start recommendations: returns { recs: [[position, score]], warnings, usedFallback }. */
export function recommendForNewUser(cat, nb, ratings, n = 10) {
  const { clean, warnings } = validateRatings(cat, ratings);
  const raw = rawScores(cat, nb, clean);
  for (let i = 0; i < raw.length; i++) if (clean.has(i) || cat.nt[i] < cat.minRatings) raw[i] = NaN;
  let recs = topN(raw, n).map((p) => [p, round2(clip(raw[p]))]);
  let usedFallback = false;
  if (!recs.length) {
    warnings.push("A personalised list was not possible, so these are popular movies instead.");
    recs = popular(cat, new Set(clean.keys()), n);
    usedFallback = true;
  }
  return { recs, warnings, usedFallback };
}

/** Most similar movies to one position: [[position, similarity]] (sim > 0 only). */
export function similarMovies(cat, nb, pos, n = 10) {
  const out = [];
  for (let j = 0; j < Math.min(n, nb.k); j++) {
    const s = nb.sim[pos * nb.k + j] / 65535;
    if (s > 0) out.push([nb.idx[pos * nb.k + j], Math.round(s * 1000) / 1000]);
  }
  return out;
}

/** "Shawshank Redemption, The (1994)" -> { title: "The Shawshank Redemption", year: "1994" } */
export function splitTitle(raw) {
  let title = raw, year = "";
  const m = raw.match(/^(.*)\s\((\d{4})(?:[-–]\d{0,4})?\)\s*$/);
  if (m) { title = m[1]; year = m[2]; }
  const art = title.match(/^(.*?), (The|A|An)((?: \(.+\))?)$/);
  if (art) title = `${art[2]} ${art[1]}${art[3]}`;
  return { title, year };
}
