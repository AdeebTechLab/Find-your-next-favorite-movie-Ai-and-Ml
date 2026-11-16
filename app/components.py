"""Pure HTML/CSS building blocks for the UI. No Streamlit import, so they are unit-testable.

Every dynamic string is HTML-escaped. HTML is emitted on ONE line because Streamlit's markdown
treats indented lines as code blocks.
"""
from __future__ import annotations

import hashlib
import html
import math
import re

import pandas as pd

ACCENT = "#e5484d"
_YEAR = re.compile(r"\((\d{4})\)\s*$")
_ARTICLE = re.compile(r"^(.*), (The|A|An)$")

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
#MainMenu, footer {visibility: hidden;}
h1, h2, h3 {letter-spacing: -0.01em;}

/* ---------- hero ---------- */
.hero {border-radius: 18px; padding: 34px 38px; margin-bottom: 22px;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(229,72,77,.35), transparent 60%),
              linear-gradient(135deg, rgba(120,119,198,.22), rgba(229,72,77,.10));
  border: 1px solid rgba(128,128,128,.25);}
.hero h1 {margin: 0 0 6px 0; font-size: 2.3rem; line-height: 1.15;}
.hero p {margin: 0; opacity: .82; font-size: 1.05rem; max-width: 760px;}
.pill {display: inline-block; padding: 3px 11px; margin: 14px 8px 0 0; border-radius: 999px; font-size: .78rem;
  border: 1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.10);}

/* ---------- KPI tiles ---------- */
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 6px 0 20px;}
.kpi {border-radius: 14px; padding: 16px 18px; border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.07);}
.kpi .label {font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65;}
.kpi .value {font-size: 1.75rem; font-weight: 700; margin-top: 2px;}
.kpi .hint {font-size: .78rem; opacity: .6; margin-top: 2px;}

/* ---------- steps ---------- */
.steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin: 8px 0 8px;}
.step {border-radius: 14px; padding: 18px; border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.06);}
.step .n {display:inline-flex; width: 28px; height: 28px; border-radius: 50%; align-items:center; justify-content:center;
  background: #e5484d; color: #fff; font-weight: 700; font-size: .85rem; margin-bottom: 8px;}
.step b {display:block; margin-bottom: 4px;}
.step span {opacity: .75; font-size: .92rem;}

/* ---------- movie cards ---------- */
.grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px; margin: 10px 0 6px;}
.grid.compact {grid-template-columns: 1fr;}
.mcard {display: flex; border-radius: 14px; overflow: hidden; border: 1px solid rgba(128,128,128,.25);
  background: rgba(128,128,128,.07); transition: transform .12s ease, border-color .12s ease;}
.mcard:hover {transform: translateY(-2px); border-color: rgba(229,72,77,.65);}
.poster {position: relative; flex: 0 0 88px; display:flex; align-items:flex-end; justify-content:flex-start; padding: 8px; color:#fff;}
.poster .rank {position:absolute; top: 8px; left: 8px; min-width: 26px; height: 26px; padding: 0 6px; box-sizing: border-box;
  display: inline-flex; align-items: center; justify-content: center; font-weight: 800; font-size: .85rem;
  border-radius: 999px; background: rgba(0,0,0,.5);}
.poster .year {font-weight: 700; font-size: 1.05rem; text-shadow: 0 1px 6px rgba(0,0,0,.5);}
.mbody {flex: 1; padding: 12px 14px; min-width: 0;}
.mtitle {font-weight: 650; font-size: 1.02rem; line-height: 1.25; margin-bottom: 6px; overflow-wrap: anywhere;}
.chips {margin-bottom: 8px;}
.chip {display:inline-block; font-size: .7rem; padding: 2px 9px; margin: 0 5px 4px 0; border-radius: 999px;
  background: hsla(var(--h), 65%, 50%, .18); border: 1px solid hsla(var(--h), 65%, 55%, .45);}
.score {display:flex; align-items: baseline; justify-content: space-between; gap: 8px;}
.score .lbl {font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; opacity: .65;}
.score .val {font-size: 1.25rem; font-weight: 750;}
.bar {height: 6px; border-radius: 999px; background: rgba(128,128,128,.25); overflow: hidden; margin: 4px 0 7px;}
.bar > i {display:block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #f2a65a, #e5484d);}
.sub {font-size: .78rem; opacity: .68;}

/* ---------- misc ---------- */
.callout {border-left: 4px solid #e5484d; padding: 10px 16px; border-radius: 6px; background: rgba(229,72,77,.08); margin: 10px 0;}
.empty {text-align:center; padding: 34px 10px; border: 1px dashed rgba(128,128,128,.4); border-radius: 14px; opacity: .8;}
.sidebrand {font-size: 1.25rem; font-weight: 800; letter-spacing: -0.01em;}
.sidesub {font-size: .8rem; opacity: .6; margin-bottom: 8px;}
</style>
"""


# ------------------------------------------------------------------ helpers
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def split_title(raw_title: str) -> tuple[str, str]:
    """'Matrix, The (1999)' -> ('The Matrix', '1999')."""
    title = str(raw_title)
    year = ""
    match = _YEAR.search(title)
    if match:
        year, title = match.group(1), title[: match.start()].strip()
    article = _ARTICLE.match(title)
    if article:
        title = f"{article.group(2)} {article.group(1)}"
    return title, year


def hue(text: str) -> int:
    """Stable 0-359 hue derived from text (same movie/genre always gets the same colour)."""
    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:6], 16) % 360


def _is_number(value) -> bool:
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def genre_chips(genres: str, limit: int = 3) -> str:
    parts = [g for g in str(genres).split("|") if g and g != "(no genres listed)"][:limit]
    return "".join(f'<span class="chip" style="--h:{hue(g)}">{esc(g)}</span>' for g in parts)


# ------------------------------------------------------------------ score formats
_SCORE_KINDS = {
    # column         label         formatter                       bar fraction
    "predicted_rating": ("Predicted", lambda v: f"★ {v:.2f}", lambda v: v / 5.0),
    "your_rating": ("You rated", lambda v: f"★ {v:.1f}", lambda v: v / 5.0),
    "similarity": ("Match", lambda v: f"{v * 100:.0f}%", lambda v: v),
}


def movie_card(row: pd.Series, rank: int | None, score_col: str) -> str:
    label, fmt, frac = _SCORE_KINDS[score_col]
    name, year = split_title(row["title"])
    h = hue(row["title"])
    poster_style = f"background: linear-gradient(145deg, hsl({h},58%,38%), hsl({(h + 40) % 360},55%,16%));"

    score = float(row[score_col]) if _is_number(row[score_col]) else None
    score_txt = fmt(score) if score is not None else "n/a"
    width = max(0.0, min(1.0, frac(score))) * 100 if score is not None else 0.0

    avg = f"{float(row['avg_rating']):.2f}" if _is_number(row.get("avg_rating")) else "n/a"
    n_ratings = int(row["n_ratings"]) if _is_number(row.get("n_ratings")) else 0
    rank_html = f'<span class="rank">{rank}</span>' if rank else ""

    return (
        '<div class="mcard">'
        f'<div class="poster" style="{poster_style}">{rank_html}<span class="year">{esc(year)}</span></div>'
        '<div class="mbody">'
        f'<div class="mtitle">{esc(name)}</div>'
        f'<div class="chips">{genre_chips(row["genres"])}</div>'
        f'<div class="score"><span class="lbl">{label}</span><span class="val">{score_txt}</span></div>'
        f'<div class="bar"><i style="width:{width:.0f}%"></i></div>'
        f'<div class="sub">Avg rating {avg} &middot; {n_ratings:,} ratings</div>'
        "</div></div>"
    )


def cards_html(df: pd.DataFrame, score_col: str, numbered: bool = True, compact: bool = False) -> str:
    """A responsive grid of cards. Empty frames produce a friendly empty state."""
    if df is None or df.empty:
        return '<div class="empty">No results to show for this request.</div>'
    cards = "".join(
        movie_card(row, i + 1 if numbered else None, score_col) for i, (_, row) in enumerate(df.iterrows())
    )
    return f'<div class="grid{" compact" if compact else ""}">{cards}</div>'


def kpi_row(items: list[tuple[str, str, str]]) -> str:
    """items = [(label, value, hint), ...]"""
    tiles = "".join(
        f'<div class="kpi"><div class="label">{esc(l)}</div><div class="value">{esc(v)}</div>'
        f'<div class="hint">{esc(h)}</div></div>'
        for l, v, h in items
    )
    return f'<div class="kpis">{tiles}</div>'


def hero(title: str, subtitle: str, pills: list[str]) -> str:
    pill_html = "".join(f'<span class="pill">{esc(p)}</span>' for p in pills)
    return f'<div class="hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p>{pill_html}</div>'


def steps_html(steps: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="step"><div class="n">{i}</div><b>{esc(t)}</b><span>{esc(d)}</span></div>'
        for i, (t, d) in enumerate(steps, 1)
    )
    return f'<div class="steps">{body}</div>'


def callout(text: str) -> str:
    return f'<div class="callout">{esc(text)}</div>'