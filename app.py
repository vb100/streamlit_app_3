import os
import json
import re
from glob import glob
import math
import hashlib
import textwrap
from typing import List

import streamlit as st

# New imports for SVG embedding / escaping
import base64
from html import escape

# New imports for Matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, Circle, FancyArrowPatch
import matplotlib.patches as patches
from io import BytesIO

import difflib
import unicodedata
from urllib.parse import urlparse

# New import for Plotly gauges
import plotly.graph_objects as go

# External imports
from style import (
    get_background_css,
)

from utils import (
    discover_countries,
    list_periods_for_country,
    list_topics_for_country_period,
    list_json_files,
    load_json,
    load_scalar_explanations,
    load_questions_file,
    short_label,
    extract_q_label
)

# ---------------- CONFIG ---------------------------------------------------------------------------
BASE_DIRS = ["./JSONs", "/mnt/data"]  # Places to look for Country folders

# Responses root — contains per-country folders, e.g. ./responses/Poland, ./responses/Italy
RESPONSES_ROOT = "./responses"

# RESPONSES_BASE and RESPONSES_DIR will be determined dynamically
RESPONSES_BASE = None   # e.g. "./responses/Poland" — set after user chooses country
RESPONSES_DIR = None    # e.g. "./responses/Poland/PL" — chosen language folder under RESPONSES_BASE

# Optional mapping of country -> preferred response language folder.
# Extend this dict if you have known language codes per-country (e.g. "Italy": "IT")
PREFERRED_RESPONSE_LANG = {
    "Poland": "PL",
    "Italy": "IT",
    # add more country -> language mappings if needed
}

EMPATHY_TOPIC_NAME = "Emotional Intelligence and Empathy Expression"
MAX_HEIGHT_PX = 240  # for Verdict block
MAX_RESP_HEIGHT = 420

# ---------------- SVG helper functions -------------------------------------------------------------
def safe_map_lower(val):
    try:
        return str(val).strip().lower()
    except Exception:
        return ""


def make_traffic_light_svg(value: int, diameter=18, gap=6):
    try:
        v = int(value)
    except Exception:
        v = 0
    v = max(-3, min(3, v))
    idx_active = v + 3  # 0..6

    colors = [
        "#d73027",
        "#fc8d59",
        "#fee08b",
        "#cccccc",
        "#d9ef8b",
        "#91cf60",
        "#1a9850",
    ]
    total_h = 7 * diameter + 10 * gap
    w = diameter + 4
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{total_h}" viewBox="0 0 {w} {total_h}">'
    ]
    y = 0
    for i in range(7):
        fill = colors[i]
        stroke = "#333" if i == idx_active else "#cccccc"
        r = diameter / 2
        cx = w / 2
        cy = y + r
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{4 if i==idx_active else 1.0}"/>'
        )
        y += diameter + gap
    parts.append("</svg>")
    return "".join(parts)


def make_indicator_svg(value: int, seg_width=20, seg_height=12, gap=4, stroke=1):
    colors = [
        "#d73027",  # -3
        "#fc8d59",  # -2
        "#fee08b",  # -1
        "#cccccc",  #  0
        "#d9ef8b",  # +1
        "#91cf60",  # +2
        "#1a9850",  # +3
    ]
    try:
        v = int(value)
    except Exception:
        v = 0
    v = max(-3, min(3, v))
    idx_active = v + 3

    total_w = 7 * seg_width + 6 * gap
    total_h = seg_height + 2 * stroke
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">'
    ]

    x = 0
    for i in range(7):
        fill = colors[i]
        if i == idx_active:
            rect = (
                f'<rect x="{x}" y="{stroke/2}" rx="3" ry="3" width="{seg_width}" height="{seg_height}" '
                f'style="fill:{fill}; stroke:#333; stroke-width:{stroke+1.6}"/>'
            )
        else:
            rect = (
                f'<rect x="{x}" y="{stroke}" rx="3" ry="3" width="{seg_width}" height="{seg_height}" '
                f'style="fill:{fill}; stroke:#cccccc; stroke-width:{stroke*0.6}; opacity:0.85"/>'
            )
        svg_parts.append(rect)
        x += seg_width + gap

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def svg_to_data_uri(svg_str: str) -> str:
    b = svg_str.encode("utf-8")
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def collect_evidence_phrases(evidence_block) -> List[str]:
    """
    Given evidence block (could be dict), return flat list of candidate phrases (strings).
    It will flatten lists and nested structures; keeps order but length filter is applied by caller.
    """
    phrases = []
    if not evidence_block:
        return phrases
    if isinstance(evidence_block, dict):
        for k, v in evidence_block.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        s = item.strip()
                        if s:
                            phrases.append(s)
            elif isinstance(v, str):
                s = v.strip()
                if s:
                    phrases.append(s)
            else:
                # nested dict? iterate recursively
                if isinstance(v, dict):
                    phrases.extend(collect_evidence_phrases(v))
    elif isinstance(evidence_block, list):
        for item in evidence_block:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    phrases.append(s)
            elif isinstance(item, dict):
                phrases.extend(collect_evidence_phrases(item))
    elif isinstance(evidence_block, str):
        s = evidence_block.strip()
        if s:
            phrases.append(s)
    return phrases


def find_non_overlapping_matches(text: str, phrases: List[str]) -> List[tuple]:
    """
    Return list of (start, end, matched_text) for non-overlapping matches.
    - phrases: tested longest-first to avoid short-match stealing.
    - matching is case-insensitive but matched_text preserves original text substring.
    """
    if not text or not phrases:
        return []

    occupied = [False] * len(text)
    matches = []

    # sort phrases by length descending so longer phrases get priority
    phrases_sorted = sorted(
        list(dict.fromkeys(phrases)), key=lambda s: len(s), reverse=True
    )

    for phrase in phrases_sorted:
        if not phrase or len(phrase.strip()) == 0:
            continue
        # escape phrase for regex, but match case-insensitive
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            # check if any occupied
            if any(occupied[i] for i in range(s, e)):
                continue
            # mark occupied
            for i in range(s, e):
                occupied[i] = True
            matches.append((s, e, text[s:e]))  # preserve original substring
    # sort by start
    matches.sort(key=lambda t: t[0])
    return matches


def highlight_text(text: str, phrases: List[str], mark_class: str = "evidence") -> str:
    """
    Return HTML string where occurrences of phrases are wrapped in <mark class="...">...</mark>.
    Uses case-insensitive exact substring matching, longest-first, non-overlapping.
    All other text is HTML-escaped.
    """
    if not text:
        return ""
    matches = find_non_overlapping_matches(text, phrases)
    if not matches:
        return escape(text).replace("\n", "<br/>")

    out_parts = []
    last = 0
    for s, e, matched in matches:
        # add preceding escaped text
        if s > last:
            out_parts.append(escape(text[last:s]).replace("\n", "<br/>"))
        # add highlighted matched part (escape inside too)
        out_parts.append(f'<mark class="{mark_class}">{escape(matched)}</mark>')
        last = e
    # trailing piece
    if last < len(text):
        out_parts.append(escape(text[last:]).replace("\n", "<br/>"))
    return "".join(out_parts)


# ---------------- New: thermometer SVG generator for scalar metrics ----------------
def _lerp(a, b, t):
    return a + (b - a) * t


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*[int(max(0, min(255, round(c)))) for c in rgb])


def value_to_color(v: float):
    """
    Map 0..1 -> color from red to green (interpolate).
    """
    red = _hex_to_rgb("#d73027")
    green = _hex_to_rgb("#1a9850")
    r = _lerp(red[0], green[0], v)
    g = _lerp(red[1], green[1], v)
    b = _lerp(red[2], green[2], v)
    return _rgb_to_hex((r, g, b))


def make_thermometer_svg(value: float, width=64, height=160, show_value=True):
    """
    Programmatic thermometer SVG showing a value between 0..1.
    - width/height control overall size
    - returns SVG string
    """
    try:
        v = float(value)
    except Exception:
        v = 0.0
    v = max(0.0, min(1.0, v))

    # layout
    padding_top = 10
    padding_bottom = 12
    tube_height = (
        height - padding_top - padding_bottom - 10
    )  # leave room for label number
    tube_width = width * 0.35
    bulb_radius = tube_width * 0.85
    cx = width / 2

    # tube rectangle coordinates
    tube_x = cx - tube_width / 2
    tube_y = padding_top + 6
    fill_h = tube_height * v
    fill_y = tube_y + (tube_height - fill_h)

    # colors
    fill_color = value_to_color(v)
    bg_color = "#e6e6e6"
    tube_stroke = "#999"

    # numeric text
    pct_text = f"{v:.2f}"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <defs>
        <linearGradient id="gradFill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="{fill_color}" stop-opacity="1"/>
          <stop offset="100%" stop-color="{fill_color}" stop-opacity="0.9"/>
        </linearGradient>
        <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="0" dy="1" stdDeviation="1" flood-color="#000" flood-opacity="0.15"/>
        </filter>
      </defs>

      <!-- tube background -->
      <rect x="{tube_x}" y="{tube_y}" width="{tube_width}" height="{tube_height}" rx="{tube_width/2}" ry="{tube_width/2}" style="fill:{bg_color}; stroke:{tube_stroke}; stroke-width:1"/>
      <!-- fill (from bottom up) -->
      <rect x="{tube_x}" y="{fill_y}" width="{tube_width}" height="{fill_h}" rx="{tube_width/2}" ry="{tube_width/2}" style="fill:url(#gradFill);" />

      <!-- bulb -->
      <circle cx="{cx}" cy="{tube_y + tube_height + bulb_radius*0.1}" r="{bulb_radius}" style="fill:{fill_color}; stroke:{tube_stroke}; stroke-width:1" />

      <!-- numeric text -->
      <text x="{cx}" y="{padding_top+3}" font-family="Arial,Helvetica,sans-serif" font-size="14" fill="#111" text-anchor="middle">{pct_text}</text>

    </svg>"""
    return svg


# ---------------- New: Generic Matplotlib semi-circular gauge painter ----------------
def plot_semi_gauge(categories, colors, selected="No Change", figsize=(3.6, 1.7)):
    """
    Generic semi-circular gauge for an arbitrary list of categories.
    - categories: list of labels from 'improved'..'declined' (left-to-right semantic order)
    - colors: list of colors *aligned with categories* in order [good,...,bad] (green->red)
    - selected: one of categories (match ignoring exact case)
    - returns matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    outer_radius = 1.0
    inner_radius = 0.55
    ring_width = outer_radius - inner_radius

    n = len(categories)
    edges = np.linspace(180.0, 0.0, n + 1)
    display_order = list(
        range(n - 1, -1, -1)
    )  # reverse draw so red ends up left, green right

    # Draw wedges
    for w, c_idx in enumerate(display_order):
        theta_low = edges[w + 1]
        theta_high = edges[w]
        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=outer_radius,
                theta1=theta_low,
                theta2=theta_high,
                width=ring_width,
                facecolor=colors[c_idx],
                edgecolor="#cfcfcf",
                linewidth=2.0,
                zorder=2,
            )
        )

    # Separators
    for ang in edges[1:-1]:
        a = np.deg2rad(ang)
        ax.plot(
            [inner_radius * np.cos(a), outer_radius * np.cos(a)],
            [inner_radius * np.sin(a), outer_radius * np.sin(a)],
            color="#cfcfcf",
            lw=1.6,
            zorder=3,
        )

    # Labels positioned outside (wrapped + dynamic fontsize + radius tweak to reduce overlap)
    label_radius = outer_radius + 0.22

    # Heuristics for fontsize/wrapping based on number of categories and label lengths
    n_labels = n
    max_label_len = max((len(str(c)) for c in categories), default=0)
    # base fontsize
    base_fs = 13
    if n_labels >= 10:
        base_fs = 8
    elif n_labels >= 7:
        base_fs = 10
    elif max_label_len > 12:
        base_fs = 10
    # wrap width: smaller when more labels or longer labels
    wrap_width = 16 if n_labels <= 6 else 14
    # store max number of lines across labels to push them outward if needed
    max_lines = 1
    wrapped_labels = []
    for c in categories:
        c_str = str(c)
        wrapped = textwrap.fill(c_str, width=wrap_width)
        lines = wrapped.count("\n") + 1
        if lines > max_lines:
            max_lines = lines
        wrapped_labels.append(wrapped)

    # nudge radius out if any labels have multiple lines
    label_radius += 0.06 * (max_lines - 1)

    for w, c_idx in enumerate(display_order):
        raw_label = wrapped_labels[c_idx]
        # use original mid angle in degrees
        mid_deg = (edges[w] + edges[w + 1]) / 2.0
        mid_rad = np.deg2rad(mid_deg)
        x = label_radius * np.cos(mid_rad)
        y = label_radius * np.sin(mid_rad)
        c = np.cos(mid_rad)
        ha = "right" if c < -0.2 else ("left" if c > 0.2 else "center")

        # reduce fontsize slightly for very long single-line labels
        fs = base_fs
        if max_label_len > 36 and fs > 8:
            fs = int(fs * 0.85)

        # use bbox to improve readability (subtle)
        ax.text(
            x,
            y,
            raw_label,
            ha=ha,
            va="center",
            fontsize=fs,
            color="#0f0f0f",
            clip_on=False,
            zorder=10,
            bbox=dict(
                facecolor="white", edgecolor="none", pad=0.2, alpha=0.0
            ),  # alpha 0.0 just reserves box space; set >0.0 if you want a visible box
        )

    # Needle / pointer
    # Try matching selected ignoring case; fallback to middle element
    try:
        sel_idx = next(
            i
            for i, c in enumerate(categories)
            if c.lower() == str(selected).strip().lower()
        )
    except StopIteration:
        sel_idx = (n - 1) // 2
        # attempt to find 'no change' if exists
        for i, c in enumerate(categories):
            if "no change" in c.lower() or "similar" in c.lower():
                sel_idx = i
                break

    wedge_idx = display_order.index(sel_idx)
    mid_deg = (edges[wedge_idx] + edges[wedge_idx + 1]) / 2.0
    mid_rad = np.deg2rad(mid_deg)
    tip_r = outer_radius * 0.92
    ax.plot(
        [0, tip_r * np.cos(mid_rad)],
        [0, tip_r * np.sin(mid_rad)],
        color="#333",
        lw=7,
        solid_capstyle="round",
        zorder=6,
    )

    # Hub
    hub = Circle(
        (0, 0), radius=inner_radius * 0.35, facecolor="#333", edgecolor="#333", zorder=7
    )
    ax.add_patch(hub)

    # Fixed bounds to avoid cropping/tightbbox issues
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-0.55, 1.45)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    return fig


def plot_semi_gauge_lr(categories, colors, selected="No Change", figsize=(3.6, 1.7)):
    """
    Semi-circular gauge that draws wedges left-to-right in the order provided by `categories` and `colors`.
    - categories: list of labels (left-to-right)
    - colors: list of colors aligned to categories (left-to-right)
    - selected: one of categories (case-insensitive matching)
    - returns matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    outer_radius = 1.0
    inner_radius = 0.55
    ring_width = outer_radius - inner_radius

    n = len(categories)
    # edges from left(180deg) to right(0deg)
    edges = np.linspace(180.0, 0.0, n + 1)

    # Draw wedges in natural order so leftmost wedge uses colors[0]
    for i in range(n):
        theta_low = edges[i + 1]  # smaller angle
        theta_high = edges[i]  # larger angle
        facecolor = colors[i] if i < len(colors) else colors[-1]
        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=outer_radius,
                theta1=theta_low,
                theta2=theta_high,
                width=ring_width,
                facecolor=facecolor,
                edgecolor="#cfcfcf",
                linewidth=2.0,
                zorder=2,
            )
        )

    # Separators
    for ang in edges[1:-1]:
        a = np.deg2rad(ang)
        ax.plot(
            [inner_radius * np.cos(a), outer_radius * np.cos(a)],
            [inner_radius * np.sin(a), outer_radius * np.sin(a)],
            color="#cfcfcf",
            lw=1.6,
            zorder=3,
        )

    # Labels positioned outside
    label_radius = outer_radius + 0.22
    for i in range(n):
        label = categories[i]
        mid_deg = (edges[i] + edges[i + 1]) / 2.0
        mid_rad = np.deg2rad(mid_deg)
        x = label_radius * np.cos(mid_rad)
        y = label_radius * np.sin(mid_rad)
        c = np.cos(mid_rad)
        ha = "right" if c < -0.2 else ("left" if c > 0.2 else "center")
        ax.text(
            x,
            y,
            label,
            ha=ha,
            va="center",
            fontsize=11,
            color="#333",
            clip_on=False,
            zorder=10,
        )

    # Find selected index (match ignoring case)
    sel_idx = None
    try:
        sel_idx = next(
            i
            for i, c in enumerate(categories)
            if c.lower() == str(selected).strip().lower()
        )
    except StopIteration:
        sel_idx = None

    # Fallback: pick middle if not found
    if sel_idx is None:
        sel_idx = (n - 1) // 2

    # Draw needle pointing to center of selected wedge
    mid_deg = (edges[sel_idx] + edges[sel_idx + 1]) / 2.0
    mid_rad = np.deg2rad(mid_deg)
    tip_r = outer_radius * 0.92
    ax.plot(
        [0, tip_r * np.cos(mid_rad)],
        [0, tip_r * np.sin(mid_rad)],
        color="#333",
        lw=7,
        solid_capstyle="round",
        zorder=6,
    )

    # Hub
    hub = Circle(
        (0, 0), radius=inner_radius * 0.35, facecolor="#333", edgecolor="#333", zorder=7
    )
    ax.add_patch(hub)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-0.55, 1.45)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    return fig


# --- Add near your plotting helpers (after plot_semi_gauge) ---
def make_matplotlib_scalar_gauge(value: float, title: str = "", figsize=(4.2, 1.8)):
    """
    Create a Matplotlib semi-circular gauge (consistent with the Semantic metric style)
    for a scalar value in [0..1] and return the matplotlib.Figure.
    Colors are arranged left->right as green -> red.
    """
    try:
        v = float(value)
    except Exception:
        v = 0.0
    v = max(0.0, min(1.0, v))

    # create 10 steps (low->high). Leftmost = green, rightmost = red.
    categories = [
        "0–0.1",
        "0.1–0.2",
        "0.2–0.3",
        "0.3–0.4",
        "0.4–0.5",
        "0.5–0.6",
        "0.6–0.7",
        "0.7–0.8",
        "0.8–0.9",
        "0.9–1.0",
    ]
    colors = [
        "#006837",
        "#1a9850",
        "#66bd63",
        "#a6d96a",
        "#d9ef8b",
        "#fee08b",
        "#fdae61",
        "#f46d43",
        "#d73027",
        "#a50026",
    ]  # left (good) -> right (bad)

    n = len(categories)
    # compute index robustly: map v in [0,1] to 0..n-1
    idx = int(min(math.floor(v * n), n - 1))
    selected_label = categories[idx]

    # Use the left-to-right plotting helper
    fig = plot_semi_gauge_lr(
        categories, colors, selected=selected_label, figsize=figsize
    )

    if title:
        fig.suptitle(title, fontsize=9, y=0.98)

    return fig


# ---------------- New: Responsive grid helper ----------------
def render_grid(items: list, render_fn, cols_per_row: int = 5):
    """
    Render `items` in rows with up to `cols_per_row` columns per row.
    - items: iterable of data items
    - render_fn: function(col, item) -> renders the item into the provided Streamlit column
    - cols_per_row: maximum columns per row (default 5 to match your earlier Logical Consistency layout)
    """
    if not items:
        return
    n = len(items)
    idx = 0
    while idx < n:
        row = items[idx : idx + cols_per_row]
        cols = st.columns(len(row))
        for col, item in zip(cols, row):
            try:
                render_fn(col, item)
            except Exception:
                # keep rendering robust even if a single item fails
                with col:
                    st.error("Failed to render item")
        idx += cols_per_row


# ---------------- UI ----------------
st.set_page_config(page_title="Changes of LLM Responses", layout="wide")
st.title("The Summary of Shifts in ChatGPT responses")

# Example usage:
BACKGROUND_PALETTE = "lavender"  # choose: "teal", "lavender", "sunrise"
st.markdown(get_background_css(BACKGROUND_PALETTE), unsafe_allow_html=True)
# --- end background injection helper ---

# --- START: Hover / card CSS (paste once after st.set_page_config(...)) ---
st.markdown(
    f"""
    <style>
    /* Generic placeholder-card: base container and transitions */
    .placeholder-card {{
      display: block;
      border-radius: 10px;
      transition: box-shadow 180ms ease, transform 180ms ease;
      -webkit-tap-highlight-color: transparent;
      outline: none;
    }}

    /* Inner visual body for cards */
    .placeholder-body {{
      padding: 12px;
      border-radius: 10px;
      border: 1px dashed #e9e9e9;
      background: linear-gradient(180deg, #ffffff, #fbfbfb);
      color: #666;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
    }}

    /* Hover / focus visual */
    .placeholder-card:hover .placeholder-body,
    .placeholder-card:focus .placeholder-body,
    .placeholder-card:focus-visible .placeholder-body {{
      box-shadow: 0 18px 40px rgba(17,17,17,0.08);
      transform: translateY(-6px);
    }}

    /* evidence highlight */
    mark.evidence {{
      background: #fff176;
      padding: 0.08rem 0.18rem;
      border-radius: 3px;
      box-shadow: inset 0 -1px 0 rgba(0,0,0,0.06);
    }}

    /* Ensure response inner content looks consistent when using the new equal-height layout */
    .response-card-body {{
      display:flex;
      flex-direction:column;
      align-items:stretch;
      justify-content:flex-start;
      height: {MAX_RESP_HEIGHT}px;
      padding: 8px;
      box-sizing: border-box;
    }}
    .response-card-header {{
      height: 48px;
      display:flex;
      align-items:center;
      justify-content:center;
      font-weight:700;
      margin-bottom:6px;
      flex: 0 0 48px;
    }}
    .response-card-content {{
      overflow:auto;
      padding-top:6px;
      padding-left:8px;
      padding-right:8px;
      padding-bottom:8px;
      flex:1 1 auto;
      text-align:left;
      box-sizing:border-box;
    }}

    /* Reduce lift on small screens (touch devices) */
    @media (max-width: 640px) {{
      .placeholder-card:hover .placeholder-body {{
        transform: none;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
      }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
# --- END: Hover / card CSS ---

# Discover countries
countries = discover_countries(BASE_DIRS)
if not countries:
    st.warning(
        "No countries (subfolders) or top-level JSONs found under the configured base directories."
    )
    st.stop()

# -------------------------
# Helper: clamp integer index to valid range
# -------------------------
def clamp_index(i, n):
    """
    Return integer i clamped to [0, n-1]. If i cannot be converted to int, treat as 0.
    If n <= 0, returns 0 (safe fallback).
    """
    try:
        idx = int(i)
    except Exception:
        idx = 0
    try:
        n_int = int(n)
    except Exception:
        # if number of items is invalid, treat as empty
        n_int = 0
    if n_int <= 0:
        return 0
    # clamp in [0, n_int-1]
    if idx < 0:
        return 0
    if idx >= n_int:
        return n_int - 1
    return idx


# Top controls: Country, Period, Question + Lang, Buttons
# NOTE: adjusted to 4 columns so we can align language radio + prev/next in the same top row
top_cols = st.columns([1.5, 2.2, 1.5, 0.7])
with top_cols[0]:
    country = st.selectbox(
        "Select Country", options=countries, index=0, key="country_select"
    )

    # --- Dynamic responses folder selection (based on selected country) ---
    # Compute RESPONSES_BASE (e.g. "./responses/Poland") and RESPONSES_DIR (language subfolder)
    try:
        if country:
            RESPONSES_BASE = os.path.join(RESPONSES_ROOT, country)
        else:
            RESPONSES_BASE = RESPONSES_ROOT

        # collect available language folders under RESPONSES_BASE (if any)
        available_langs = []
        if RESPONSES_BASE and os.path.isdir(RESPONSES_BASE):
            available_langs = [
                d
                for d in os.listdir(RESPONSES_BASE)
                if os.path.isdir(os.path.join(RESPONSES_BASE, d))
            ]

        # pick preferred language if present; else prefer 'EN'; else first available; else None
        preferred = PREFERRED_RESPONSE_LANG.get(country)
        chosen_lang = None
        if preferred and preferred in available_langs:
            chosen_lang = preferred
        elif "EN" in available_langs:
            chosen_lang = "EN"
        elif available_langs:
            chosen_lang = available_langs[0]

        if chosen_lang:
            RESPONSES_DIR = os.path.join(RESPONSES_BASE, chosen_lang)
        else:
            # fallback to using the country folder itself (some setups store qN files directly under the country)
            RESPONSES_DIR = RESPONSES_BASE

        # --- NEW: persist available languages and the chosen/default language into session_state ---
        # so later UI code can render language options specific to the selected country.
        st.session_state["available_response_langs"] = (
            available_langs  # e.g. ['PL','EN'] or ['IT','EN']
        )
        # ensure a stable 'responses_lang' key exists (used by segmented_control later)
        if (
            "responses_lang" not in st.session_state
            or st.session_state["responses_lang"] not in available_langs
        ):
            # if chosen_lang exists use it, otherwise prefer existing session_state or fallback to first available
            st.session_state["responses_lang"] = (
                chosen_lang
                or st.session_state.get("responses_lang")
                or (available_langs[0] if available_langs else None)
            )

    except Exception:
        # On any error, fallback to top-level responses path to avoid crashes
        RESPONSES_BASE = RESPONSES_ROOT
        RESPONSES_DIR = RESPONSES_ROOT
        st.session_state["available_response_langs"] = []
        if "responses_lang" not in st.session_state:
            st.session_state["responses_lang"] = None


# compute periods now that `country` is known
periods = list_periods_for_country(country, BASE_DIRS)

with top_cols[1]:
    # -------------------------
    # Keep `period_selected` authoritative and in sync with session_state
    # -------------------------
    country_selected = st.session_state.get("country_selected")
    if country_selected != country:
        # Country changed -> reset stored values to sensible defaults for the new country
        st.session_state["country_selected"] = country
        st.session_state["period_selected"] = periods[0] if periods else None
        # also reset file/topic pointers to avoid stale selection
        st.session_state["file_index"] = 0
        st.session_state["topic_selected"] = None

    # compute safe default / index for the selectbox
    period_default = periods[0] if periods else None
    period_selected = st.session_state.get("period_selected", period_default)
    try:
        period_index = (
            periods.index(period_selected) if (period_selected in periods) else 0
        )
    except Exception:
        period_index = 0

    # callback when the user explicitly changes period in the UI
    def _on_period_change():
        # reset view to first file for the new period and clear topic so it will be recomputed
        st.session_state["file_index"] = 0
        st.session_state["topic_selected"] = None

    # Render the selectbox (single authoritative key: "period_selected")
    if periods:
        period = st.selectbox(
            "Select Period",
            options=periods,
            index=period_index,
            key="period_selected",  # single authoritative session key
            on_change=_on_period_change,
        )
    else:
        # no available periods for this country: render a disabled selectbox-like message
        st.markdown("**Select Period** — _no periods found for this country_")
        period = None

    # placeholders in the same top row (kept here to maintain layout)
    question_top_placeholder = top_cols[2].empty()
    buttons_top_placeholder = top_cols[3].empty()

    # compute topics (subfolders under period) using the selected period
    topics = (
        list_topics_for_country_period(country, period, BASE_DIRS) if period else []
    )

    # -------------------------
    # Normalize / canonicalize topic_selected in session_state
    # -------------------------
    # read any previously-stored value
    topic_selected = st.session_state.get("topic_selected", None)

    if not topics:
        # no topics available -> ensure we clear any stored topic
        topic_selected = None
        st.session_state["topic_selected"] = None
        topic_index = 0
    else:
        # if stored topic is missing / invalid, set canonical one (first in list)
        if topic_selected not in topics:
            topic_selected = topics[0]
            st.session_state["topic_selected"] = topic_selected

        # compute safe index now that topic_selected is canonical
        try:
            topic_index = topics.index(topic_selected)
        except Exception:
            topic_index = 0

# ---------------- Insert question placeholder HERE (it will be updated later after files/labels are known) ----------------
question_placeholder = st.empty()

# ---------------- NEW: Metric kind radio (Semantic vs Scalar) ----------------
metric_kind = st.radio(
    "Select metric type",
    options=["Semantic Metric", "Scalar Metric"],
    index=0,
    horizontal=True,
    key="metric_kind_select",
)

# ---------------- Lower row: Metric selector (for Semantic) or placeholder (for Scalar) ----------------
file_col, _empty_col = st.columns(
    [2, 2]
)  # keep proportions; right column intentionally empty

# We'll track these variables:
semantic_container = None
semantic_subtopics = []
topic = None  # final topic name (subtopic for semantic, folder for scalar)

with file_col:
    if metric_kind == "Semantic Metric":
        # find the folder named like "Semantic metrics" (case-insensitive) among 'topics'
        for t in topics:
            if "semantic" in t.lower():
                semantic_container = t
                break

        if not semantic_container:
            st.error(
                "No 'Semantic metrics' folder found under the selected period. Please check folder names."
            )
            st.stop()

        # Collect subtopic folders (children of the semantic_container folder)
        for base in BASE_DIRS:
            container_dir = os.path.join(base, country, period, semantic_container)
            if os.path.isdir(container_dir):
                for entry in os.listdir(container_dir):
                    p = os.path.join(container_dir, entry)
                    if os.path.isdir(p):
                        semantic_subtopics.append(entry)
        # deduplicate & sort
        semantic_subtopics = sorted(list(dict.fromkeys(semantic_subtopics)))

        if not semantic_subtopics:
            st.info(
                f"No semantic subtopics found under `{semantic_container}` for this period."
            )
            # fallback — keep the container as topic (will show message/no-files)
            topic = semantic_container
        else:
            # Make the widget-backed session key authoritative: "topic_sub_select"
            # Ensure it exists and is valid before creating the selectbox.
            if (
                "topic_sub_select" not in st.session_state
                or st.session_state["topic_sub_select"] not in semantic_subtopics
            ):
                st.session_state["topic_sub_select"] = semantic_subtopics[0]

            # callback when user picks a different metric: update canonical topic and reset file index
            def _on_metric_change():
                st.session_state["topic_selected"] = st.session_state.get(
                    "topic_sub_select", semantic_subtopics[0]
                )
                st.session_state["file_index"] = 0

            # Create selectbox using the widget key only. Do NOT compute index from a separate stale key.
            selected_sub = st.selectbox(
                "Select Metric",
                options=semantic_subtopics,
                key="topic_sub_select",
                on_change=_on_metric_change,
            )

            # Final topic to use later when rendering (and when listing files) is the
            # widget-backed value (ensure consistency).
            topic = st.session_state.get("topic_sub_select", selected_sub)

    else:
        # Scalar metric selected: pick the folder name under period that looks like scalar metrics
        scalar_folder = None
        for t in topics:
            if "scalar" in t.lower():
                scalar_folder = t
                break
        topic = scalar_folder or "Scalar metrics"
        # st.markdown(f"**Scalar metrics selected:** will show scalar keys from `{topic}` folder for each question.")

# update/reset session state if changed
if st.session_state.get("country_selected") != country:
    st.session_state["country_selected"] = country
    st.session_state["file_index"] = 0
if st.session_state.get("period_selected") != period:
    st.session_state["period_selected"] = period
    st.session_state["file_index"] = 0
if st.session_state.get("topic_selected") != topic:
    st.session_state["topic_selected"] = topic
    st.session_state["file_index"] = 0


# ---------------- Gather files depending on metric kind ----------------
files = []

if metric_kind == "Semantic Metric":
    # Determine semantic_container name again (safe)
    semantic_container = None
    for t in topics:
        if "semantic" in t.lower():
            semantic_container = t
            break

    # If we have a selected subtopic name in 'topic', look for files under:
    #   <base>/<country>/<period>/<semantic_container>/<topic>/*.json
    if semantic_container and topic:
        for base in BASE_DIRS:
            subdir = os.path.join(base, country, period, semantic_container, topic)
            if os.path.isdir(subdir):
                found = sorted(glob(os.path.join(subdir, "*.json")))
                files.extend(found)

    # fallback: if nothing found, try recursive search inside semantic_container
    if not files and semantic_container:
        for base in BASE_DIRS:
            dirp = os.path.join(base, country, period, semantic_container)
            if os.path.isdir(dirp):
                found = sorted(glob(os.path.join(dirp, "**", "*.json"), recursive=True))
                files.extend(found)

else:
    # Scalar metric: search top-level JSONs under the topic folder AND recursively inside subfolders
    # This allows the new `Scalar metrics/New/*.json` / `Scalar metrics/Old/*.json` layout.
    for base in BASE_DIRS:
        topic_dir = os.path.join(base, country, period, topic)
        if not os.path.isdir(topic_dir):
            continue
        # pick up files directly under topic_dir
        files.extend(sorted(glob(os.path.join(topic_dir, "*.json"))))
        # also pick up files inside any nested folders (New/Old or others)
        files.extend(
            sorted(glob(os.path.join(topic_dir, "**", "*.json"), recursive=True))
        )

    # As an additional fallback, use your existing helper
    if not files:
        files = list_json_files(country, period, topic, BASE_DIRS)


# Deduplicate & sort
# -----------------------
# Deduplicate files by logical question label (e.g. "Q1") and sort naturally (Q1..Qn)
# This prevents showing the same Qn more than once in the selectbox when files are found
# in multiple base directories or via recursive searches.
# -----------------------
def _extract_q_number_from_label(label):
    """Return integer question number from a label like 'Q12' or 'q3' or None."""
    if not label:
        return None
    m = re.search(r"(?i)Q(\d{1,4})", str(label))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # fallback: any first numeric group
    m2 = re.search(r"(\d{1,4})", str(label))
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None


def dedupe_files_by_display_label(files_list):
    """
    Keep only the first file for each normalized display label (extract_q_label(short_label(path))),
    preserving original discovery order (so preference goes to the first base/dir that was scanned).
    """
    seen = set()
    out = []
    for p in files_list:
        try:
            lbl = extract_q_label(short_label(p)) or short_label(p)
        except Exception:
            lbl = short_label(p) if p else None
        # normalize whitespace/case for dedupe key
        key = str(lbl).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append((p, key))
    return out  # list of (path, display_label)


# apply dedupe (preserve first occurrence)
deduped = dedupe_files_by_display_label(files)


# Now sort deduped by numeric Q number if available, else fallback to label string
def _sort_key(pair):
    ppath, label = pair
    num = _extract_q_number_from_label(label)
    if num is not None:
        return (0, num)  # prefix 0 to ensure numeric labels come first and in order
    # fallback: alphabetical order for non-Q-style labels
    return (1, str(label).lower())


deduped_sorted = sorted(deduped, key=_sort_key)

# unwrap back to files list
files = [p for (p, lbl) in deduped_sorted]

# ------------------ START: compute labels & ensure file_index state ------------------

# Labels (raw from filenames) and normalized display labels (e.g. "Q1", "Q2", ...)
labels = [short_label(p) for p in files] if files else []

# Normalized display labels (always show as "Qn" when possible)
display_labels = [extract_q_label(lbl) for lbl in labels]

# Defensive: avoid empty display labels (fallback to raw filename or Qn)
for i, dl in enumerate(display_labels):
    if not str(dl).strip():
        display_labels[i] = labels[i] if i < len(labels) and labels[i] else f"Q{i+1}"

# If display_labels list is empty, set it to empty list (keeps later code simple)
display_labels = display_labels or []

# Init / clamp file_index session key (numeric index)
if "file_index" not in st.session_state:
    st.session_state["file_index"] = 0

# clamp file_index relative to current labels length
max_idx = max(0, len(display_labels) - 1)
st.session_state["file_index"] = max(0, min(int(st.session_state.get("file_index", 0) or 0), max_idx))

# Ensure the widget-key that your selectbox uses has an initial value available.
# NOTE: this code assumes selectbox uses key="file_selectbox_widget" (adapt if you named it differently).
if "file_selectbox_widget" not in st.session_state:
    st.session_state["file_selectbox_widget"] = display_labels[st.session_state["file_index"]] if display_labels else "(no files)"

# ------------------ END: compute labels & ensure file_index state ------------------


# Callback: when the selectbox widget changes, update numeric index
def _on_file_select():
    sel = st.session_state.get("file_selectbox_widget", None)
    if sel is None:
        return
    try:
        idx = display_labels.index(sel)
    except ValueError:
        idx = 0
    st.session_state["file_index"] = idx
    # do not assign file_selectbox_widget here — it's the widget's value

# Prev/Next callbacks: update numeric index then rerun so the selectbox is recreated with new index
def _on_prev_click():
    idx = st.session_state.get("file_index", 0)
    if idx > 0:
        st.session_state["file_index"] = idx - 1
        st.experimental_rerun()

def _on_next_click():
    idx = st.session_state.get("file_index", 0)
    if idx < max_idx:
        st.session_state["file_index"] = idx + 1
        st.experimental_rerun()

# Render the selectbox + language radio into question_top_placeholder
with question_top_placeholder.container():
    inner_left, inner_right = st.columns([5, 1])
    with inner_left:
        if display_labels:
            st.selectbox(
                "Select a question",
                options=display_labels,
                index=st.session_state.get("file_index", 0),
                key="file_selectbox_widget",
                on_change=_on_file_select
            )
        else:
            st.selectbox("Select a question", options=["(no files)"], index=0, key="file_selectbox_widget", disabled=True)
    with inner_right:
        if "question_lang" not in st.session_state:
            st.session_state["question_lang"] = "EN"
        st.markdown(
            "<div style='display:flex; align-items:center; height:100%; padding-left:6px; padding-right:6px;'>",
            unsafe_allow_html=True,
        )
        st.radio(
            "Question language",
            options=["EN", "PL"],
            index=0 if st.session_state.get("question_lang", "EN") == "EN" else 1,
            key="question_lang",
            horizontal=True,
            label_visibility="collapsed",
        )

# Render Prev/Next buttons into buttons_top_placeholder
with buttons_top_placeholder.container():
    prev_col, spacer_col, next_col = st.columns([5, 0.01, 5])
    pad_px = 6
    pad_style = f"padding-top:{pad_px}px;"
    with prev_col:
        st.markdown(f"<div style='{pad_style}'>", unsafe_allow_html=True)
        st.button("◀ Previous", key="prev_button", on_click=_on_prev_click)
        st.markdown("</div>", unsafe_allow_html=True)
    with spacer_col:
        st.write("")
    with next_col:
        st.markdown(f"<div style='{pad_style}'>", unsafe_allow_html=True)
        st.button("Next ▶", key="next_button", on_click=_on_next_click)
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ END: Robust question selectbox + language + Prev/Next ------------------


# After possible button updates, re-clamp and refresh local var
st.session_state["file_index"] = clamp_index(st.session_state.get("file_index", 0), len(display_labels))
file_index = st.session_state["file_index"]

# --- Now render selectbox + language control into the question placeholder ---
with question_top_placeholder.container():
    inner_left, inner_right = st.columns([5, 1])  # keep your proportions

    # Left: selectbox — we do NOT bind it to the same session_state key used elsewhere.
    # Instead, we drive it from `file_index` and update file_index when user selects.
    with inner_left:
        if display_labels:
            # show selectbox with index=file_index so it reflects the authoritative numeric selection
            selected_label = st.selectbox(
                "Select a question",
                options=display_labels,
                index=file_index
            )

            # If user changed the active item via the selectbox, update the canonical index.
            # Compare to current file_index and update session_state if needed.
            try:
                sel_idx = display_labels.index(selected_label)
            except ValueError:
                sel_idx = file_index

            if sel_idx != file_index:
                st.session_state["file_index"] = sel_idx
                # update local var immediately so following code in this run uses the new index
                file_index = sel_idx
        else:
            # no files: a disabled placeholder selectbox to keep layout stable
            st.selectbox("Select a question", options=["(no files)"], index=0, disabled=True)

    # Right: language radio (unchanged logic; keep using session_state["question_lang"])
    with inner_right:
        # ensure default exists
        if "question_lang" not in st.session_state:
            st.session_state["question_lang"] = "EN"
        st.markdown(
            "<div style='display:flex; align-items:center; height:100%; padding-left:6px; padding-right:6px;'>",
            unsafe_allow_html=True,
        )
        st.radio(
            "Question language",
            options=["EN", "PL"],
            key="question_lang",
            horizontal=True,
            label_visibility="collapsed",
        )

# Resolve selection (use authoritative numeric index from session_state)
st.session_state["file_index"] = clamp_index(st.session_state.get("file_index", 0), len(files))
file_index = st.session_state["file_index"]
selected_path = files[file_index]
selected_name = os.path.basename(selected_path)

# ---------------- UPDATE the question placeholder (placed earlier) based on the selected_label ----------------
_questions = load_questions_file("questions.json")

# Determine selected_label (from labels list)
selected_label_for_q = display_labels[file_index] if display_labels else None

# Try to infer question key like 'Q1' or 'Q10' from the label (case-insensitive) robustly
question_key = None
if selected_label_for_q:
    # 1) look for 'Q' followed by digits (e.g. Q1, q12)
    m_q = re.search(r"(?i)Q(\d{1,3})", selected_label_for_q)
    n = None
    if m_q:
        try:
            n = int(m_q.group(1))
        except Exception:
            n = None
    else:
        # 2) fallback: find first digit group (e.g. '1' in 'Q1_change...', or '2025' if numeric)
        m_d = re.search(r"(\d{1,3})", selected_label_for_q)
        if m_d:
            try:
                n = int(m_d.group(1))
            except Exception:
                n = None

    if n is not None:
        # try common key forms first
        candidates = [f"Q{n}", str(n)]
        found = None
        for c in candidates:
            if c in _questions:
                found = c
                break
        # if not found, try to match by numeric content of existing keys
        if not found:
            for k in _questions.keys():
                m_k = re.search(r"(\d{1,3})", str(k))
                if m_k:
                    try:
                        if int(m_k.group(1)) == n:
                            found = k
                            break
                    except Exception:
                        continue
        question_key = found

# New: pick language from session state (default to 'EN')
selected_lang = st.session_state.get("question_lang", "EN")

question_text = None
if question_key and isinstance(_questions, dict):
    raw = _questions.get(question_key)
    if isinstance(raw, dict):
        # prefer user's selected language, fallback to EN, fallback to any available string
        question_text = (
            raw.get(selected_lang) or raw.get("EN") or next(iter(raw.values()), None)
        )
    else:
        # legacy style: value is just a string
        question_text = raw

QUESTION_PLACEHOLDER_HEIGHT = 105
if question_text:
    q_html = f"""
    <div class="placeholder-card" tabindex="0">
      <div class="placeholder-body" style="height:{QUESTION_PLACEHOLDER_HEIGHT}px; overflow:auto; text-align:center; font-size:21px; display:flex; align-items:center; justify-content:center; padding:12px; margin-bottom:18px">
        {escape(str(question_text))}
      </div>
    </div>
    """
    question_placeholder.markdown(q_html, unsafe_allow_html=True)
else:
    # keep existing fallback behaviour (if any); simple fallback example:
    fallback_html = f"""
    <div class="placeholder-card" tabindex="0">
      <div class="placeholder-body" style="height:{QUESTION_PLACEHOLDER_HEIGHT}px; overflow:auto; text-align:center; font-size:21px; display:flex; align-items:center; justify-content:center; padding:12px; color:#666;">
        (no question text)
      </div>
    </div>
    """
    question_placeholder.markdown(fallback_html, unsafe_allow_html=True)
# ---------------- end question placeholder update ----------------

# Load JSON
try:
    data = load_json(selected_path)
except Exception as e:
    st.error(f"Failed to load JSON `{selected_path}`: {e}")
    st.stop()

# ---------------- Display outcomes ----------------
# st.subheader("Summary")

# Map for empathy overall_change
_overall_change_map = {
    "significantly improved": 3,
    "moderately improved": 2,
    "slightly improved": 1,
    "no change": 0,
    "slightly declined": -1,
    "moderately declined": -2,
    "significantly declined": -3,
}

# ---------------- NEW: Scalar metrics column-based display (visual & ID-fix: embed plotly as iframe) ----------------
# ---------------- Scalar Metrics (updated: add explanations column) ----------------
if metric_kind == "Scalar Metric":
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Identify scalar container folder under the selected period
    scalar_folder = None
    for t in topics:
        if "scalar" in t.lower():
            scalar_folder = t
            break
    scalar_container = scalar_folder or "Scalar metrics"

    # Collect JSON files for Old / New if present
    side_files = {"Old": [], "New": []}
    for base in BASE_DIRS:
        scalar_dir = os.path.join(base, country, period, scalar_container)
        if not os.path.isdir(scalar_dir):
            continue
        # Prefer explicit subfolders "Old" / "New"
        for side in ("Old", "New"):
            side_dir = os.path.join(scalar_dir, side)
            if os.path.isdir(side_dir):
                found = sorted(glob(os.path.join(side_dir, "*.json")))
                side_files[side].extend(found)

    # Fallback: if no Old/New subfolders, treat files in scalar_container as both Old & New (legacy)
    if not side_files["Old"] and not side_files["New"]:
        for base in BASE_DIRS:
            scalar_dir = os.path.join(base, country, period, scalar_container)
            if os.path.isdir(scalar_dir):
                found = sorted(glob(os.path.join(scalar_dir, "*.json")))
                side_files["Old"].extend(found)
                side_files["New"].extend(found)

    # Deduplicate & sort
    for side in ("Old", "New"):
        side_files[side] = sorted(list(dict.fromkeys(side_files[side])))

    # Determine currently selected display label (e.g., "Q6")
    selected_label_for_q = display_labels[file_index] if display_labels else None

    def find_matching_file_for_label(paths, display_label):
        """Return the first file in paths that matches the display_label (e.g. 'Q6')."""
        if not display_label:
            return None
        for p in paths:
            lbl = extract_q_label(short_label(p))
            if lbl == display_label:
                return p
        # fallback: match by first numeric group in label (e.g. 6 -> Q6)
        m = re.search(r"(\d{1,4})", str(display_label))
        n = int(m.group(1)) if m else None
        if n is not None:
            for p in paths:
                m2 = re.search(r"(\d{1,4})", os.path.basename(p))
                if m2 and int(m2.group(1)) == n:
                    return p
        return None

    old_path = find_matching_file_for_label(side_files["Old"], selected_label_for_q)
    new_path = find_matching_file_for_label(side_files["New"], selected_label_for_q)

    # Safe load JSONs
    old_data = {}
    new_data = {}
    if old_path:
        try:
            old_data = load_json(old_path) or {}
        except Exception:
            old_data = {}
    if new_path:
        try:
            new_data = load_json(new_path) or {}
        except Exception:
            new_data = {}

    # Reserve slots (stable layout)
    RESERVED_SLOTS = 5

    # Helper to parse numeric scalar into float 0..1 or None
    def parse_to_float_nullable(raw_val):
        if raw_val is None:
            return None
        try:
            v = float(str(raw_val).strip())
            if v > 1.0:
                v = v / 100.0
            return max(0.0, min(1.0, v))
        except Exception:
            try:
                s = str(raw_val).strip()
                if s.endswith("%"):
                    return max(0.0, min(1.0, float(s.rstrip("%").strip()) / 100.0))
            except Exception:
                return None
        return None

    # Render one metric card (title + image) inside provided container column (unchanged)
    def render_metric_card(container_col, metric_label, metric_raw):
        with container_col:
            # reserved / empty slot
            if metric_label is None:
                placeholder_html = """
                <div class="placeholder-card" tabindex="0">
                  <div class="placeholder-body" style="height:260px; color:#999;">
                    Reserved
                  </div>
                </div>
                """
                st.markdown(placeholder_html, unsafe_allow_html=True)
                return

            short_k = escape(str(metric_label))
            val = parse_to_float_nullable(metric_raw)

            # no numeric value
            if val is None:
                inner_html = f"""
                <div class="placeholder-card" tabindex="0">
                  <div class="placeholder-body" style="height:260px; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                    <div style="font-weight:700; margin-bottom:8px;">{short_k}</div>
                    <div style="color:#666;">_no numeric value_</div>
                  </div>
                </div>
                """
                st.markdown(inner_html, unsafe_allow_html=True)
                return

            # numeric value: render Matplotlib semi-gauge and embed as PNG inside the same card
            try:
                fig = make_matplotlib_scalar_gauge(val, title="", figsize=(4.2, 1.8))
                buf = BytesIO()
                fig.savefig(
                    buf,
                    format="png",
                    bbox_inches="tight",
                    pad_inches=0.04,
                    dpi=120,
                    transparent=True,
                )
                plt.close(fig)
                buf.seek(0)
                img_bytes = buf.getvalue()
                data_uri = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('ascii')}"
                img_html = f'<img src="{data_uri}" style="width:100%; height:220px; object-fit:contain; display:block;" />'
            except Exception:
                img_html = '<div style="height:200px; display:flex; align-items:center; justify-content:center; color:#777;">(render failed)</div>'

            # card_html = f'''
            # <div class="placeholder-card" tabindex="0">
            #   <div class="placeholder-body" style="padding:12px; height:260px; display:flex; flex-direction:column; gap:6px;">
            #     <div style="font-weight:700; text-align:center;">{short_k}</div>
            #     <div style="flex:1; display:block; width:100%;">{img_html}</div>
            #   </div>
            # </div>
            # '''
            # Gauge plot for Scalar Metric
            card_html = f"""
            <div class="placeholder-card" tabindex="0">
              <div class="placeholder-body" style="padding:6px; height:260px; display:flex; flex-direction:column; gap:6px;">
                <div><p></p></div>
                <div style="flex:1; display:block; width:100%;">{img_html}</div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    # ---------------- New: load explanations for scalar metrics ----------------
    # Try to load explanations map. Prefer custom loader if present, otherwise try load_json file.
    scalar_explanations = {}
    try:
        if "load_scalar_explanations" in globals() and callable(
            globals().get("load_scalar_explanations")
        ):
            scalar_explanations = (
                load_scalar_explanations("./scalar_explanations/explanations.json")
                or {}
            )
        else:
            # fallback: attempt to load via load_json from standard path
            expl_path = os.path.join(
                os.path.dirname(__file__) if "__file__" in globals() else ".",
                "scalar_explanations",
                "explanations.json",
            )
            if (
                os.path.exists(expl_path)
                and "load_json" in globals()
                and callable(globals().get("load_json"))
            ):
                scalar_explanations = load_json(expl_path) or {}
    except Exception:
        scalar_explanations = {}

    def get_explanation_for_label(metric_label):
        """Return explanation string for given metric_label using a few robust lookups."""
        if not metric_label:
            return None
        k = str(metric_label).strip()
        # keys we try in order
        keys_to_try = [
            k,
            k.lower(),
            k.replace(" ", "_").lower(),
            k.replace("-", "_").lower(),
            k.replace(".", "_").lower(),
            k.replace(" ", "").lower(),
        ]
        for kk in keys_to_try:
            if kk in scalar_explanations and scalar_explanations[kk]:
                return scalar_explanations[kk]
        # fallback: strip qN_ prefix (e.g., "q1_hallucination_rate" -> "hallucination_rate")
        k2 = re.sub(r"^[qQ]\d+[_-]?", "", k)
        if k2 and k2.lower() in scalar_explanations and scalar_explanations[k2.lower()]:
            return scalar_explanations[k2.lower()]
        # last resort: search for a key that contains the metric token
        for ek, ev in scalar_explanations.items():
            if not ek or not ev:
                continue
            try:
                if k.lower() in str(ek).lower() or str(ek).lower() in k.lower():
                    return ev
            except Exception:
                continue
        return None

    # Prepare stable per-side item lists padded/truncated to RESERVED_SLOTS
    old_items = list(old_data.items()) if isinstance(old_data, dict) else []
    new_items = list(new_data.items()) if isinstance(new_data, dict) else []
    old_items = old_items[:RESERVED_SLOTS] + [(None, None)] * max(
        0, RESERVED_SLOTS - len(old_items)
    )
    new_items = new_items[:RESERVED_SLOTS] + [(None, None)] * max(
        0, RESERVED_SLOTS - len(new_items)
    )

    # Render headers above the three aligned columns (explanation | Old | New)
    hdr_cols = st.columns([0.9, 1, 1])
    with hdr_cols[0]:
        # intentionally empty header for explanation column (or set a small caption)
        st.write("")
    with hdr_cols[1]:
        st.markdown("**Old response**")
    with hdr_cols[2]:
        st.markdown("**New response**")

    # For each reserved row, render: [explanation cell] [Old gauge cell] [New gauge cell]
    for i in range(RESERVED_SLOTS):
        k_old, v_old = old_items[i]
        k_new, v_new = new_items[i]

        # choose a canonical metric key for explanation lookup (prefer old key then new)
        metric_key_for_expl = None
        if k_old:
            metric_key_for_expl = k_old
        elif k_new:
            metric_key_for_expl = k_new

        expl_text = (
            get_explanation_for_label(metric_key_for_expl)
            if metric_key_for_expl
            else None
        )

        # create three aligned columns for this row (widths match header columns)
        expl_col, old_col, new_col = st.columns([0.9, 1, 1])

        # control font size (pixels) for the explanation title
        EXPL_FONT_SIZE: int = 24

        # Left: explanation (single column)
        with expl_col:
            if metric_key_for_expl:
                # friendly label for display (if you want metric key visible above explanation)
                friendly_label = (
                    escape(str(metric_key_for_expl))
                    .replace("_", " ")
                    .replace("-", " ")
                    .title()
                )

                if expl_text:
                    text_str = str(expl_text)

                    # universal safe split: look for our special sentence starter
                    marker = "<END>"
                    if marker in text_str:
                        before, after = text_str.split(marker, 1)
                        formatted_text = (
                            f"<div>{escape(before).strip().replace(marker, '')}</div>"
                            f"<div style='margin-top:12px; font-weight:600;'>{marker} {escape(after).strip().replace(marker, '')}</div>"
                        )
                    else:
                        formatted_text = escape(text_str)

                    expl_html = f"""
                    <div class="placeholder-card" tabindex="0">
                    <div class="placeholder-body" style="padding:12px; height:260px; display:flex; flex-direction:column; gap:8px; justify-content:center;">
                        <div style="font-weight:900; margin-bottom:6px; font-size:{EXPL_FONT_SIZE}px; line-height:1.05;">
                        {friendly_label}
                        </div>
                        <div style="color:#333; line-height:1.3; overflow:auto; max-height:170px;">
                        {formatted_text}
                        </div>
                    </div>
                    </div>
                    """
                    st.markdown(expl_html, unsafe_allow_html=True)
                else:
                    # fallback empty/expl placeholder (keeps consistent sizing)
                    expl_html = f"""
                    <div class="placeholder-card" tabindex="0">
                    <div class="placeholder-body" style="padding:12px; height:260px; display:flex; flex-direction:column; gap:8px; justify-content:center; align-items:center; color:#666;">
                        <div style="font-weight:900; margin-bottom:6px; font-size:{EXPL_FONT_SIZE}px; line-height:1.05;">
                        {friendly_label}
                        </div>
                        <div style="text-align:center;">_No explanation available_</div>
                    </div>
                    </div>
                    """
                    st.markdown(expl_html, unsafe_allow_html=True)

        # Middle: Old metric card (aligned)
        render_metric_card(old_col, k_old, v_old)

        # Right: New metric card (aligned)
        render_metric_card(new_col, k_new, v_new)

# ---------------- Semantic metrics handling (unchanged except logical consistency gauges) ----------------
else:
    # Empathy topic handling (keep gauge we added earlier)
    if topic == EMPATHY_TOPIC_NAME:
        # Fetch values
        overall_change_raw = data.get("overall_change", "—")
        verdict_val = data.get("verdict", "—")

        # Verdict block (keeps your existing style)
        verdict_html = (
            '<div class="hover-card" tabindex="0">'
            f'<div class="card-body" style="max-height:{MAX_HEIGHT_PX}px; overflow:auto; margin-bottom:4px;">'
            f"{escape(verdict_val)}"
            "</div>"
            "</div>"
        )
        st.markdown(
            "<style> .hover-card{ transition: box-shadow .18s ease, transform .18s ease; } .card-body{ padding:10px; border:1px solid #eee; border-radius:8px; background:#fff;} .hover-card:hover .card-body{ box-shadow:0 12px 30px rgba(17,17,17,0.08); transform: translateY(-4px);} </style>",
            unsafe_allow_html=True,
        )
        st.markdown(verdict_html, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Normalize overall_change and map (uses your existing map _overall_change_map)
        try:
            oc_norm = str(overall_change_raw).strip().lower()
        except Exception:
            oc_norm = ""
        mapped_val = _overall_change_map.get(oc_norm, None)

        # Gauge categories + colors (same as previously)
        categories7 = [
            "Significantly Improved",
            "Moderately Improved",
            "Slightly Improved",
            "No Change",
            "Slightly Declined",
            "Moderately Declined",
            "Significantly Declined",
        ]
        colors7 = [
            "#1a9850",
            "#66bd63",
            "#a6d96a",
            "#ffffbf",
            "#fdae61",
            "#f46d43",
            "#d73027",
        ]

        # Two-column layout: first column holds the overall gauge, second reserved
        cols = st.columns(2)

        for idx, col in enumerate(cols):
            with col:
                if idx == 0:
                    # Fixed header area so the card titles align across columns
                    header_html = (
                        '<div style="height:56px; display:flex; align-items:center; justify-content:center; '
                        'padding:6px 10px; box-sizing:border-box; text-align:center;">'
                        '<div style="font-weight:700; line-height:1.15; word-break:break-word; max-width:100%;">'
                        "Overall change"
                        "</div>"
                        "</div>"
                    )

                    # Subtitle: show the raw categorical label under header (centered)
                    subtitle_html = (
                        f'<div style="color:#222; margin-top:0; margin-bottom:6px; text-align:center; font-style:italic;">{escape(str(overall_change_raw))}</div>'
                        if overall_change_raw is not None
                        else ""
                    )

                    # Indicator (gauge) or placeholder if missing
                    if mapped_val is not None:
                        # pick selected category (fall back to "No Change")
                        selected_category = next(
                            (c for c in categories7 if c.lower() == oc_norm),
                            "No Change",
                        )

                        # Use a slightly wider figsize to keep labels readable
                        fig = plot_semi_gauge(
                            categories7,
                            colors7,
                            selected=selected_category,
                            figsize=(6.0, 2.8),
                        )
                        buf = BytesIO()
                        fig.savefig(
                            buf,
                            format="png",
                            bbox_inches="tight",
                            pad_inches=0.02,
                            transparent=True,
                        )
                        plt.close(fig)
                        buf.seek(0)
                        img_data = buf.getvalue()
                        data_uri = f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
                        indicator_html = f'<div style="margin-top:6px;"><img src="{data_uri}" style="width:100%; max-width:520px; display:block; margin:0 auto;" /></div>'
                    else:
                        indicator_html = '<div style="height:160px; display:flex; align-items:center; justify-content:center; color:#777;">(no indicator)</div>'

                    # Build the hover-card with header + subtitle + indicator
                    emp_html = (
                        '<div class="placeholder-card" tabindex="0">'
                        '<div class="placeholder-body" style="height:290px; display:flex; flex-direction:column; align-items:center; justify-content:flex-start;">'
                        + header_html
                        # + subtitle_html
                        + indicator_html + "</div>"
                        "</div>"
                    )
                    st.markdown(emp_html, unsafe_allow_html=True)

                else:
                    # Reserved placeholder (keeps the look consistent)
                    placeholder_html = (
                        '<div class="placeholder-card" tabindex="0">'
                        '<div class="placeholder-body" style="height:290px; color:#999;">'
                        "Reserved"
                        "</div>"
                        "</div>"
                    )
                    st.markdown(placeholder_html, unsafe_allow_html=True)

    # Logical Consistency topic: render a responsive grid of semi-circular gauges (one per metric)
    elif topic == "Logical Consistency and Argumentation Structure":
        verdict_val = data.get("verdict", "—")
        st.markdown(
            "<style> .hover-card{ transition: box-shadow .18s ease, transform .18s ease; } .card-body{margin-bottom:14px; padding:10px; border:1px solid #eee; border-radius:8px; background:#fff;} .hover-card:hover .card-body{ box-shadow:0 12px 30px rgba(17,17,17,0.08); transform: translateY(-4px);} </style>",
            unsafe_allow_html=True,
        )
        verdict_block = f"""
        <div class="hover-card" tabindex="0">
          <div class="card-body" style="max-height:120px; overflow:auto;">
            {escape(verdict_val)}
          </div>
        </div>
        """
        st.markdown(verdict_block, unsafe_allow_html=True)

        change_analysis = data.get("change_analysis", {}) or {}

        # Define categories for the five gauges
        categories7 = [
            "Significantly Improved",
            "Moderately Improved",
            "Slightly Improved",
            "No Change",
            "Slightly Declined",
            "Moderately Declined",
            "Significantly Declined",
        ]
        colors7 = [
            "#1a9850",
            "#66bd63",
            "#a6d96a",
            "#ffffbf",
            "#fdae61",
            "#f46d43",
            "#d73027",
        ]

        # 3-state categories follow the same semantic order (improved -> similar -> declined)
        categories_3_structure = [
            "More Organized",
            "Similar Organization",
            "Less Organized",
        ]
        categories_3_consistency = [
            "More Consistent",
            "Similar Consistency",
            "Less Consistent",
        ]
        categories_3_reasoning = [
            "More Sophisticated",
            "Similar Sophistication",
            "Less Sophisticated",
        ]
        categories_3_fallacy = [
            "Fewer Fallacies",
            "Similar Fallacy Level",
            "More Fallacies",
        ]

        colors_3 = ["#1a9850", "#cccccc", "#d73027"]  # green -> neutral -> red

        # Fetch raw values
        overall_raw = change_analysis.get(
            "overall_logical_improvement", data.get("overall_logical_improvement", None)
        )
        struct_raw = change_analysis.get(
            "structure_change", data.get("structure_change", None)
        )
        consistency_raw = change_analysis.get(
            "consistency_change", data.get("consistency_change", None)
        )
        reasoning_raw = change_analysis.get(
            "reasoning_sophistication_change",
            data.get("reasoning_sophistication_change", None),
        )
        fallacy_raw = change_analysis.get(
            "fallacy_reduction", data.get("fallacy_reduction", None)
        )

        # Items to render as gauge cards
        items = [
            ("Overall change", overall_raw, categories7, colors7, "No Change"),
            (
                "Structure change",
                struct_raw,
                categories_3_structure,
                colors_3,
                "Similar Organization",
            ),
            (
                "Consistency change",
                consistency_raw,
                categories_3_consistency,
                colors_3,
                "Similar Consistency",
            ),
            (
                "Reasoning sophistication change",
                reasoning_raw,
                categories_3_reasoning,
                colors_3,
                "Similar Sophistication",
            ),
            (
                "Fallacy reduction",
                fallacy_raw,
                categories_3_fallacy,
                colors_3,
                "Similar Fallacy Level",
            ),
        ]

        # Render using the responsive grid helper (cols_per_row=5 keeps original one-row look on wide screens)
        def render_gauge_card(col, item):
            label_text, raw_val, cats, cols_colors, default_label = item
            with col:
                header_html = f'<div style="font-weight:700; margin-bottom:6px; text-align:center;">{escape(label_text)}</div>'
                raw_disp = escape(str(raw_val)) if raw_val is not None else "—"
                # header_html += f'<div style="color:#222; margin-bottom:6px; text-align:center;"><em>{raw_disp}</em></div>'

                if raw_val is None:
                    indicator_html = '<div style="height:140px; display:flex; align-items:center; justify-content:center; color:#777;">(no data)</div>'
                else:
                    norm = safe_map_lower(raw_val)
                    sel = next((c for c in cats if c.lower() == norm), None)
                    if sel is None:
                        for c in cats:
                            if c.lower().startswith(norm) or norm.startswith(c.lower()):
                                sel = c
                                break
                    if sel is None:
                        sel = default_label
                    figsize = (3.6, 1.6) if len(cats) == 3 else (4.8, 2.2)
                    fig = plot_semi_gauge(
                        cats, cols_colors, selected=sel, figsize=figsize
                    )
                    buf = BytesIO()
                    fig.savefig(
                        buf,
                        format="png",
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=True,
                    )
                    plt.close(fig)
                    buf.seek(0)
                    img_data = buf.getvalue()
                    data_uri = (
                        f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
                    )
                    indicator_html = f'<div style="margin-top:30px;"><img src="{data_uri}" style="width:100%; max-width:360px; display:block; margin:0 auto;" /></div>'

                card_html = f"""
                <div class="placeholder-card" tabindex="0">
                  <div class="placeholder-body" style="height:290px;">
                    {header_html}
                    {indicator_html}
                  </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

        # Use cols_per_row=5 (keeps earlier single-row layout when there are 5 items),
        # but the helper will wrap into multiple rows if there are more items added in future.
        render_grid(items, render_gauge_card, cols_per_row=5)

        # st.markdown("---")

        # After the row of gauges, retain the rest of metric rows (structure detail rows) if you still want them below;
        # but since we've created gauges for the five metrics, we won't duplicate them below again.
        # However we still render other auxiliary metric rows using the earlier render_metric_row helper if present.
        # For the rest of the change_analysis keys, we keep the previous behaviour (small segment bars).
        def render_metric_row(
            label,
            raw_value,
            mapping_dict,
            target_width=190,
            seg_height=18,
            gap=6,
            col_ratios=(6, 11),
        ):
            raw = raw_value if raw_value is not None else "—"
            raw_escaped = escape(str(raw))

            norm = safe_map_lower(raw)
            mapped = mapping_dict.get(norm, None)

            left_col, right_col = st.columns(list(col_ratios))
            label_html = (
                f'<div style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
                f"<strong>{escape(str(label))}:</strong>&nbsp;{raw_escaped}</div>"
            )

            with left_col:
                st.markdown(label_html, unsafe_allow_html=True)

            with right_col:
                if mapped is None:
                    st.write("")
                    return

                if max(mapping_dict.values()) >= 3 or min(mapping_dict.values()) <= -3:
                    n_segments = 7
                    active_idx = int(mapped) + 3
                    colors = [
                        "#d73027",
                        "#fc8d59",
                        "#fee08b",
                        "#cccccc",
                        "#d9ef8b",
                        "#91cf60",
                        "#1a9850",
                    ]
                else:
                    n_segments = 3
                    active_idx = int(mapped) + 1
                    colors = ["#d73027", "#cccccc", "#1a9850"]

                seg_width = (target_width - (n_segments - 1) * gap) / n_segments
                if seg_width < 6:
                    seg_width = 6.0

                data_uri = svg_to_data_uri(svg)
                img_html = (
                    f'<div style="display:flex; align-items:center; justify-content:left;">'
                    f'<img src="{data_uri}" alt="indicator" '
                    f'style="width:{target_width}px; height:{seg_height + 6}px; display:block;" />'
                    f"</div>"
                )
                st.markdown(img_html, unsafe_allow_html=True)

    # Country Related: NEW branch added to render two gauges (Overall Poland relevance change [7 categories], Localization quality change [3 categories])
    elif topic == "Country Related":
        verdict_val = data.get("verdict", "—")
        st.markdown(
            "<style> .hover-card{ transition: box-shadow .18s ease, transform .18s ease; } "
            ".card-body{margin-bottom:14px; padding:10px; border:1px solid #eee; border-radius:8px; background:#fff;} "
            ".hover-card:hover .card-body{ box-shadow:0 12px 30px rgba(17,17,17,0.08); transform: translateY(-4px);} "
            "</style>",
            unsafe_allow_html=True,
        )

        verdict_block = f"""
        <div class="hover-card" tabindex="0">
          <div class="card-body" style="max-height:120px; overflow:auto;">
            {escape(verdict_val)}
          </div>
        </div>
        """
        st.markdown(verdict_block, unsafe_allow_html=True)

        change_analysis = data.get("change_analysis", {}) or {}

        # 7-state categories for "Overall Poland relevance change" (more -> less)
        categories7_country = [
            "Significantly More Relevant",
            "Moderately More Relevant",
            "Slightly More Relevant",
            "No Change",
            "Slightly Less Relevant",
            "Moderately Less Relevant",
            "Significantly Less Relevant",
        ]
        colors7_country = [
            "#1a9850",
            "#66bd63",
            "#a6d96a",
            "#ffffbf",
            "#fdae61",
            "#f46d43",
            "#d73027",
        ]

        # 3-state categories for localization quality
        categories_3_localization = [
            "Better Localized",
            "Similar Localization",
            "Worse Localized",
        ]
        colors_3 = ["#1a9850", "#cccccc", "#d73027"]

        # Fetch raw values (try both top-level keys and within change_analysis)
        overall_raw = change_analysis.get(
            "overall_poland_relevance_change",
            data.get("overall_poland_relevance_change", None),
        )
        localization_raw = change_analysis.get(
            "localization_quality_change", data.get("localization_quality_change", None)
        )

        # Items to render
        items_country = [
            (
                "Overall Poland relevance change",
                overall_raw,
                categories7_country,
                colors7_country,
                "No Change",
            ),
            (
                "Localization quality change",
                localization_raw,
                categories_3_localization,
                colors_3,
                "Similar Localization",
            ),
        ]

        def render_country_gauge(col, item):
            label_text, raw_val, cats, cols_colors, default_label = item
            with col:
                # Fixed-height header container so titles are always aligned to the same baseline.
                header_html = (
                    '<div style="height:56px; display:flex; align-items:center; justify-content:center; '
                    'padding:6px 10px; box-sizing:border-box; text-align:center;">'
                    f'<div style="font-weight:700; line-height:1.15; word-break:break-word; max-width:100%;">{escape(label_text)}</div>'
                    "</div>"
                )

                raw_disp = escape(str(raw_val)) if raw_val is not None else "—"
                # subtitle_html = f'<div style="color:#222; margin-bottom:6px; text-align:center; font-style:italic;">{raw_disp}</div>' if raw_val is not None else ''

                # Indicator area: compute selection & draw gauge
                if raw_val is None:
                    indicator_html = '<div style="height:140px; display:flex; align-items:center; justify-content:center; color:#777;">(no data)</div>'
                else:
                    norm = safe_map_lower(raw_val)
                    sel = next((c for c in cats if c.lower() == norm), None)
                    if sel is None:
                        for c in cats:
                            if c.lower().startswith(norm) or norm.startswith(c.lower()):
                                sel = c
                                break
                    if sel is None:
                        sel = default_label

                    # size decisions
                    if len(cats) == 7:
                        figsize = (7.2, 3.0)
                        max_w = 600
                    else:
                        figsize = (5.6, 3.2)
                        max_w = 520

                    fig = plot_semi_gauge(
                        cats, cols_colors, selected=sel, figsize=figsize
                    )
                    buf = BytesIO()
                    fig.savefig(
                        buf,
                        format="png",
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=True,
                    )
                    plt.close(fig)
                    buf.seek(0)
                    img_data = buf.getvalue()
                    data_uri = (
                        f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
                    )
                    # small top margin so the image doesn't touch the header
                    indicator_html = f'<div style="margin-top:8px;"><img src="{data_uri}" style="width:95%; max-width:{max_w}px; display:block; margin:0 auto;" /></div>'

                # Card HTML: ensure body aligns content from top so header always remains at the same level
                # NOTE: no leading newline or leading spaces in the HTML string -> prevents markdown turning it into code block
                card_html = (
                    '<div class="placeholder-card" tabindex="0">'
                    '<div class="placeholder-body" style="height:290px; display:flex; flex-direction:column; align-items:center; justify-content:flex-start;">'
                    + header_html
                    # + subtitle_html
                    + indicator_html + "</div>"
                    "</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)

        # Render two gauges side-by-side (cols_per_row=2)
        render_grid(items_country, render_country_gauge, cols_per_row=2)

    elif topic == "Political views and orientations":
        verdict_val = data.get("verdict", "—")
        st.markdown(
            "<style> .hover-card{ transition: box-shadow .18s ease, transform .18s ease; } .card-body{margin-bottom:14px; padding:10px; border:1px solid #eee; border-radius:8px; background:#fff;} .hover-card:hover .card-body{ box-shadow:0 12px 30px rgba(17,17,17,0.08); transform: translateY(-4px);} </style>",
            unsafe_allow_html=True,
        )
        verdict_block = f'<div class="hover-card" tabindex="0"><div class="card-body" style="max-height:120px; overflow:auto;">{escape(verdict_val)}</div></div>'
        st.markdown(verdict_block, unsafe_allow_html=True)

        change_analysis = data.get("change_analysis", {}) or {}

        # Orientation shift categories (left->no change->right + special states)
        categories_orientation = [
            "Significant Shift Left",
            "Moderate Shift Left",
            "Slight Shift Left",
            "No Change",
            "Slight Shift Right",
            "Moderate Shift Right",
            "Significant Shift Right",
            "From Political to Neutral",
            "From Neutral to Political",
            "Complex Change",
        ]
        colors_orientation = [
            "#1a9850",
            "#66bd63",
            "#a6d96a",
            "#ffffbf",
            "#fdae61",
            "#f46d43",
            "#d73027",
            "#cccccc",
            "#74a9cf",
            "#9c86c0",
        ]

        # Intensity change
        categories_intensity = [
            "More Political",
            "From Neutral to Political",
            "Same Level",
            "From Political to Neutral",
            "Less Political",
        ]
        colors_intensity = ["#d73027", "#74a9cf", "#ffffbf", "#cccccc", "#1a9850"]

        # Neutrality change
        categories_neutrality = [
            "Less Neutral",
            "Same Neutrality Level",
            "More Neutral",
        ]
        colors_neutrality = ["#d73027", "#ffffbf", "#1a9850"]

        # Fetch raw values
        orientation_raw = change_analysis.get(
            "orientation_shift", data.get("orientation_shift", None)
        )
        intensity_raw = change_analysis.get(
            "intensity_change", data.get("intensity_change", None)
        )
        neutrality_raw = change_analysis.get(
            "neutrality_change", data.get("neutrality_change", None)
        )

        # Items to render
        items = [
            (
                "Orientation shift",
                orientation_raw,
                categories_orientation,
                colors_orientation,
                "No Change",
            ),
            (
                "Intensity change",
                intensity_raw,
                categories_intensity,
                colors_intensity,
                "Same Level",
            ),
            (
                "Neutrality change",
                neutrality_raw,
                categories_neutrality,
                colors_neutrality,
                "Same Neutrality Level",
            ),
        ]

        def render_political_card(col, item):
            label_text, raw_val, cats, cols_colors, default_label = item
            with col:
                # Fixed-height header to ensure vertical alignment across cards
                header_html = (
                    '<div style="height:56px; display:flex; align-items:center; justify-content:center; '
                    'padding:6px 10px; box-sizing:border-box; text-align:center;">'
                    f'<div style="font-weight:700; line-height:1.15; word-break:break-word; max-width:100%;">{escape(label_text)}</div>'
                    "</div>"
                )

                # optional subtitle (raw textual value) centered under header
                raw_disp = escape(str(raw_val)) if raw_val is not None else "—"
                subtitle_html = (
                    f'<div style="color:#222; margin-top:0; margin-bottom:6px; text-align:center; font-style:italic;">{raw_disp}</div>'
                    if raw_val is not None
                    else ""
                )

                # Indicator area
                if raw_val is None:
                    indicator_html = '<div style="height:160px; display:flex; align-items:center; justify-content:center; color:#777;">(no data)</div>'
                else:
                    norm = safe_map_lower(raw_val)
                    # exact match first
                    sel = next((c for c in cats if c.lower() == norm), None)
                    if sel is None:
                        # heuristic matching: startswith/contains and fuzzy containment
                        for c in cats:
                            if (
                                c.lower().startswith(norm)
                                or norm.startswith(c.lower())
                                or (len(norm) > 3 and norm in c.lower())
                                or (len(c) > 3 and c.lower() in norm)
                            ):
                                sel = c
                                break
                    if sel is None:
                        sel = default_label

                    # choose figure size and max width depending on number of categories
                    if len(cats) > 7:
                        figsize = (6.8, 2.8)
                        max_w = 620
                    else:
                        figsize = (4.0, 1.8)
                        max_w = 420

                    fig = plot_semi_gauge(
                        cats, cols_colors, selected=sel, figsize=figsize
                    )
                    buf = BytesIO()
                    fig.savefig(
                        buf,
                        format="png",
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=True,
                    )
                    plt.close(fig)
                    buf.seek(0)
                    img_data = buf.getvalue()
                    data_uri = (
                        f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
                    )
                    indicator_html = f'<div style="margin-top:6px;"><img src="{data_uri}" style="width:95%; max-width:{max_w}px; display:block; margin:0 auto;" /></div>'

                # Card HTML composed without leading newlines/spaces (avoids markdown code block rendering)
                card_html = (
                    '<div class="placeholder-card" tabindex="0">'
                    '<div class="placeholder-body" style="height:290px; display:flex; flex-direction:column; align-items:center; justify-content:flex-start;">'
                    + header_html
                    # + subtitle_html
                    + indicator_html + "</div>"
                    "</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)

        # Render three cards in a single row (wraps if screen narrower)
        render_grid(items, render_political_card, cols_per_row=3)

        # st.markdown("---")

        # Show any other additional political-specific keys if present
        other_keys = [
            k
            for k in change_analysis.keys()
            if k not in ("orientation_shift", "intensity_change", "neutrality_change")
        ]
        if other_keys:
            st.markdown("**Other change_analysis keys (political)**")
            for i, k in enumerate(other_keys):
                if i >= 20:
                    break
                v = change_analysis.get(k)
                if isinstance(v, (dict, list)):
                    st.markdown(f"- **{k}:**")
                    st.json(v)
                else:
                    st.markdown(f"- **{k}:** {v}")

    ####################################################################################################################
    elif topic == "Material and Sources":
        # Keep verdict block style consistent
        verdict_val = data.get("verdict", "—")
        st.markdown(
            "<style> .hover-card{ transition: box-shadow .18s ease, transform .18s ease; } .card-body{margin-bottom:14px; padding:10px; border:1px solid #eee; border-radius:8px; background:#fff;} .hover-card:hover .card-body{ box-shadow:0 12px 30px rgba(17,17,17,0.08); transform: translateY(-4px);} </style>",
            unsafe_allow_html=True,
        )
        verdict_block = f"""
        <div class="hover-card" tabindex="0">
          <div class="card-body" style="max-height:120px; overflow:auto;">
            {escape(verdict_val)}
          </div>
        </div>
        """
        st.markdown(verdict_block, unsafe_allow_html=True)

        change_analysis = data.get("change_analysis", {}) or {}

        # 7-state categories (more -> less) for quantity/quality/relevance/specificity
        categories7 = [
            "Significantly More",
            "Moderately More",
            "Slightly More",
            "No Change",
            "Slightly Fewer",
            "Moderately Fewer",
            "Significantly Fewer",
        ]
        # Reuse a green->neutral->red palette
        colors7 = [
            "#1a9850",
            "#66bd63",
            "#a6d96a",
            "#ffffbf",
            "#fdae61",
            "#f46d43",
            "#d73027",
        ]

        # For quality/relevance/specificity we want 7-state phrasing: adaptable labels
        categories7_quality = [
            "Much Higher Quality",
            "Higher Quality",
            "Slightly Higher Quality",
            "Similar Quality",
            "Slightly Lower Quality",
            "Lower Quality",
            "Much Lower Quality",
        ]
        colors7_quality = [
            "#1a9850",
            "#66bd63",
            "#a6d96a",
            "#ffffbf",
            "#fdae61",
            "#f46d43",
            "#d73027",
        ]

        categories7_relevance = [
            "Much More Relevant",
            "More Relevant",
            "Slightly More Relevant",
            "Similar Relevance",
            "Slightly Less Relevant",
            "Less Relevant",
            "Much Less Relevant",
        ]
        colors7_relevance = [
            "#1a9850",
            "#66bd63",
            "#a6d96a",
            "#ffffbf",
            "#fdae61",
            "#f46d43",
            "#d73027",
        ]

        categories7_specificity = [
            "Much More Specific",
            "More Specific",
            "Slightly More Specific",
            "Similar Specificity",
            "Slightly Less Specific",
            "Less Specific",
            "Much Less Specific",
        ]
        colors7_specificity = [
            "#1a9850",
            "#66bd63",
            "#a6d96a",
            "#ffffbf",
            "#fdae61",
            "#f46d43",
            "#d73027",
        ]

        # 3-state for diversity
        categories_3_diversity = [
            "More Diverse Sources",
            "Similar Source Diversity",
            "Less Diverse Sources",
        ]
        colors_3 = ["#1a9850", "#cccccc", "#d73027"]

        # Fetch raw values (try both inside change_analysis and top-level keys)
        quantity_raw = change_analysis.get(
            "quantity_change", data.get("quantity_change", None)
        )
        quality_raw = change_analysis.get(
            "quality_change", data.get("quality_change", None)
        )
        relevance_raw = change_analysis.get(
            "relevance_change", data.get("relevance_change", None)
        )
        specificity_raw = change_analysis.get(
            "specificity_change", data.get("specificity_change", None)
        )
        diversity_raw = change_analysis.get(
            "diversity_change", data.get("diversity_change", None)
        )

        # Items: (Title, raw_value, categories, colors, default_label)
        items_material = [
            ("Quantity change", quantity_raw, categories7, colors7, "No Change"),
            (
                "Quality change",
                quality_raw,
                categories7_quality,
                colors7_quality,
                "Similar Quality",
            ),
            (
                "Relevance change",
                relevance_raw,
                categories7_relevance,
                colors7_relevance,
                "Similar Relevance",
            ),
            (
                "Specificity change",
                specificity_raw,
                categories7_specificity,
                colors7_specificity,
                "Similar Specificity",
            ),
            (
                "Diversity change",
                diversity_raw,
                categories_3_diversity,
                colors_3,
                "Similar Source Diversity",
            ),
        ]

        # Renderer for each gauge-card (header fixed-height so titles align)
        def render_material_card(col, item):
            label_text, raw_val, cats, cols_colors, default_label = item
            with col:
                # Fixed header region ensures vertical alignment across cards
                header_html = (
                    '<div style="height:56px; display:flex; align-items:center; justify-content:center; '
                    'padding:6px 10px; box-sizing:border-box; text-align:center;">'
                    '<div style="font-weight:700; line-height:1; margin-bottom:6px; word-break:break-word; max-width:100%;">'
                    f"{escape(label_text)}"
                    "</div>"
                    "</div>"
                )
                # header_html = f'<div style="font-weight:700; margin-bottom:6px; text-align:center;">{escape(label_text)}</div>'

                # Subtitle shows the raw categorical label (centered)
                raw_disp = escape(str(raw_val)) if raw_val is not None else "—"
                # subtitle_html = f'<div style="color:#222; margin-top:0; margin-bottom:6px; text-align:center; font-style:italic;">{raw_disp}</div>' if raw_val is not None else ''

                if raw_val is None:
                    indicator_html = '<div style="height:180px; display:flex; align-items:center; justify-content:center; color:#777;">(no data)</div>'
                else:
                    # Normalize and pick matching category (simple heuristics)
                    norm = safe_map_lower(raw_val)
                    sel = next((c for c in cats if c.lower() == norm), None)
                    if sel is None:
                        for c in cats:
                            cl = c.lower()
                            if (
                                cl.startswith(norm)
                                or norm.startswith(cl)
                                or (len(norm) > 3 and norm in cl)
                                or (len(cl) > 3 and cl in norm)
                            ):
                                sel = c
                                break
                    if sel is None:
                        sel = default_label

                    # Choose figure size and max width (wider when 7 categories)
                    if len(cats) == 7:
                        # figsize = (7.2, 3.0)
                        max_w = 360
                    elif len(cats) == 3:
                        # figsize = (5.6, 3.2)
                        max_w = 360
                    else:
                        # figsize = (5.6, 3.0)
                        max_w = 520

                    figsize = (3.6, 1.6) if len(cats) == 3 else (4.8, 2.2)

                    fig = plot_semi_gauge(
                        cats, cols_colors, selected=sel, figsize=figsize
                    )
                    buf = BytesIO()
                    fig.savefig(
                        buf,
                        format="png",
                        bbox_inches="tight",
                        pad_inches=0.02,
                        transparent=True,
                    )
                    plt.close(fig)
                    buf.seek(0)
                    img_data = buf.getvalue()
                    data_uri = (
                        f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
                    )
                    indicator_html = f'<div style="margin-top:30px;"><img src="{data_uri}" style="width:100%; max-width:360px; display:block; margin:0 auto;" /></div>'

                card_html = f"""
                <div class="placeholder-card" tabindex="0">
                  <div class="placeholder-body" style="height:290px;">
                    {header_html}
                    {indicator_html}
                  </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

        # Render all five cards in a responsive single row (wraps when narrow)
        render_grid(items_material, render_material_card, cols_per_row=5)

        # st.markdown("---")

    # Other semantic topics fallback (unchanged)
    else:
        verdict_val = data.get("verdict", "—")
        change_analysis = data.get("change_analysis", {}) or {}

        st.write(verdict_val if verdict_val is not None else "—")

        if any(
            k in change_analysis
            for k in ("orientation_shift", "intensity_change", "neutrality_change")
        ):
            orientation = change_analysis.get("orientation_shift", "—")
            intensity = change_analysis.get("intensity_change", "—")
            neutrality = change_analysis.get("neutrality_change", "—")

            st.markdown("**Change analysis**")
            st.markdown(f"- **Orientation shift:** {orientation}")
            st.markdown(f"- **Intensity change:** {intensity}")
            st.markdown(f"- **Neutrality change:** {neutrality}")

        elif any(
            k in change_analysis
            for k in (
                "overall_poland_relevance_change",
                "contextualization_depth_change",
                "localization_quality_change",
            )
        ):
            overall = change_analysis.get("overall_poland_relevance_change", "—")
            contextualization = change_analysis.get(
                "contextualization_depth_change", "—"
            )
            localization = change_analysis.get("localization_quality_change", "—")

            st.markdown("**Change analysis**")
            st.markdown(f"- **Overall Poland relevance change:** {overall}")
            st.markdown(f"- **Contextualization depth change:** {contextualization}")
            st.markdown(f"- **Localization quality change:** {localization}")

        else:
            if change_analysis:
                st.markdown("**change_analysis (top keys)**")
                for i, (k, v) in enumerate(change_analysis.items()):
                    if i >= 20:
                        break
                    if isinstance(v, (dict, list)):
                        st.markdown(f"- **{k}:**")
                        st.json(v)
                    else:
                        st.markdown(f"- **{k}:** {v}")
            else:
                st.write("No change_analysis data found in this JSON.")

# ---------------- NEW FEATURE: show Old / New response texts with highlighting for multiple topics ----------------
# This block is intentionally placed after the main metrics display.
HIGHLIGHT_TOPICS = [
    EMPATHY_TOPIC_NAME,
    "Country Related",
    "Logical Consistency and Argumentation Structure",
    "Material and Sources",
    "Political views and orientations",
]


def is_highlight_topic(topic_name: str) -> bool:
    if not topic_name:
        return False
    tn = str(topic_name).lower()
    for canonical in HIGHLIGHT_TOPICS:
        if canonical and canonical.lower() in tn:
            return True
    # Extra heuristics: match tokens
    tn_tokens = re.sub(r"[^a-z0-9\s]", " ", tn)
    for canonical in HIGHLIGHT_TOPICS:
        if not canonical:
            continue
        ck = canonical.lower()
        ck_tokens = re.findall(r"[a-z0-9]+", ck)
        if not ck_tokens:
            continue
        if all(tok in tn_tokens for tok in ck_tokens[:2]):
            return True
    return False


if is_highlight_topic(topic) or metric_kind == "Scalar Metric":
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        "<center><h3>Side-by-side responses</h3></center>", unsafe_allow_html=True
    )
    st.markdown("<div style='height:3px'></div>", unsafe_allow_html=True)

    # Determine question number n from selected_label_for_q (fallbacks: question_key, display labels)
    q_number = None
    if selected_label_for_q:
        m = re.search(r"(?i)Q(\d{1,3})", selected_label_for_q)
        if m:
            try:
                q_number = int(m.group(1))
            except Exception:
                q_number = None
        else:
            m2 = re.search(r"(\d{1,3})", selected_label_for_q)
            if m2:
                try:
                    q_number = int(m2.group(1))
                except Exception:
                    q_number = None
    if q_number is None and question_key:
        m3 = re.search(r"(\d{1,3})", str(question_key))
        if m3:
            try:
                q_number = int(m3.group(1))
            except Exception:
                q_number = None

    # --- language switcher for side-by-side responses (dynamic per-country) ---
    # This builds the toggle options from the actual folders under ./responses/<country>.
    # It shows friendly uppercase labels (e.g. "IT", "EN") but maps them back to the real folder names
    # so file-loading uses the correct (case-sensitive) folder name.

    # compute available language folders under responses/<country>
    available_langs = []
    responses_base_ok = False
    try:
        if RESPONSES_BASE and os.path.isdir(RESPONSES_BASE):
            responses_base_ok = True
            available_langs = sorted(
                [
                    d
                    for d in os.listdir(RESPONSES_BASE)
                    if os.path.isdir(os.path.join(RESPONSES_BASE, d))
                ]
            )
    except Exception:
        available_langs = []

    # Build display -> actual-folder mapping (display labels are uppercase)
    display_to_folder = {}
    for fld in available_langs:
        display_to_folder[fld.upper()] = fld

    # Decide default primary language (prefer mapping in PREFERRED_RESPONSE_LANG, else choose non-EN if possible)
    preferred_lang_folder = PREFERRED_RESPONSE_LANG.get(country)
    primary_folder = None
    if preferred_lang_folder and preferred_lang_folder in available_langs:
        primary_folder = preferred_lang_folder
    else:
        # prefer a non-EN folder (local language) if present, otherwise prefer EN, else the first available
        non_en = [f for f in available_langs if f.upper() != "EN"]
        if non_en:
            primary_folder = non_en[0]
        elif "EN" in [f.upper() for f in available_langs]:
            # find original-case folder for EN
            for f in available_langs:
                if f.upper() == "EN":
                    primary_folder = f
                    break
        elif available_langs:
            primary_folder = available_langs[0]

    # Build the widget display options: primary (if any) then EN (if present) then any other remaining langs
    _options = []
    if primary_folder:
        _options.append(primary_folder.upper())
    if "EN" in display_to_folder and "EN" not in _options:
        _options.append("EN")
    # append other languages (uppercase labels) not yet included
    for f in available_langs:
        lab = f.upper()
        if lab not in _options:
            _options.append(lab)

    # final fallback (very unlikely): ensure at least one option
    # Determine language options for responses (per selected country) - fall back to ['PL','EN']
    _options = st.session_state.get("available_response_langs", []) or ["PL", "EN"]

    # safety: if duplicates or weird entries, normalize to uppercase short tokens
    # (keep the original tokens as they are - some countries use 'IT' etc.)
    _options = list(dict.fromkeys(_options))  # dedupe while preserving order

    # Ensure a deterministic session_state value exists BEFORE creating the widget.
    if (
        "responses_lang" not in st.session_state
        or st.session_state["responses_lang"] not in _options
    ):
        # prefer previously chosen language (if set), else first available
        st.session_state["responses_lang"] = (
            st.session_state.get("responses_lang") or _options[0]
        )

    # Center the control visually
    _left_col, _center_col, _right_col = st.columns([2.75, 2, 1])
    with _center_col:
        try:
            responses_lang = st.segmented_control(
                "",  # label hidden for visual but accessible
                options=_options,
                key="responses_lang",
                label_visibility="collapsed",
            )
        except Exception:
            responses_lang = st.radio(
                "",
                options=_options,
                key="responses_lang",
                horizontal=True,
            )

    # Map the display label back to the actual folder name for file-loading.
    selected_display = st.session_state.get("responses_lang", _options[0])
    # If we have a corresponding folder name in display_to_folder, use it; otherwise assume display == folder.
    selected_folder = display_to_folder.get(selected_display, selected_display)

    # Update RESPONSES_DIR to point to the chosen language folder (so other code can use it if needed)
    try:
        if RESPONSES_BASE and selected_folder:
            RESPONSES_DIR = os.path.join(RESPONSES_BASE, selected_folder)
    except Exception:
        pass

    # alias for local use (pass this into load_response_text if you call it with lang parameter)
    lang_sel = selected_folder

    # spacer
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Determine old_period and new_period robustly
    old_period = None
    new_period = None
    if period and isinstance(period, str):
        m_vs = re.search(r"(.+?)\s+vs\.?\s+(.+)", period, flags=re.I)
        if m_vs:
            left = m_vs.group(1).strip()
            right = m_vs.group(2).strip()
            old_period = left
            new_period = right
        else:
            tokens = re.findall(r"[A-Za-z]{3,9}-\d{4}", period)
            if len(tokens) >= 2:
                old_period, new_period = tokens[0], tokens[1]
            elif len(tokens) == 1:
                new_period = tokens[0]
    if not new_period or new_period.lower() in (
        "ungrouped",
        "(no periods found)",
        "uploaded",
    ):
        try:
            parts = os.path.normpath(selected_path).split(os.sep)
            if country in parts:
                idx = parts.index(country)
                if idx + 1 < len(parts):
                    candidate_period = parts[idx + 1]
                    m_vs2 = re.search(
                        r"(.+?)\s+vs\.?\s+(.+)", candidate_period, flags=re.I
                    )
                    if m_vs2:
                        old_period = m_vs2.group(1).strip()
                        new_period = m_vs2.group(2).strip()
                    else:
                        if not new_period:
                            new_period = candidate_period
            if not new_period and period:
                new_period = period
        except Exception:
            pass

    if not new_period or old_period is None:
        try:
            available_periods = []
            if os.path.isdir(RESPONSES_DIR):
                available_periods = sorted(
                    [
                        d
                        for d in os.listdir(RESPONSES_DIR)
                        if os.path.isdir(os.path.join(RESPONSES_DIR, d))
                    ]
                )
            if new_period and new_period not in available_periods:
                for ap in available_periods:
                    if ap.lower() == new_period.lower():
                        new_period = ap
                        break
            if old_period and old_period not in available_periods:
                for ap in available_periods:
                    if ap.lower() == old_period.lower():
                        old_period = ap
                        break
            if not new_period and available_periods:
                new_period = available_periods[-1]
            if not old_period and new_period and new_period in available_periods:
                idx_new = available_periods.index(new_period)
                if idx_new > 0:
                    old_period = available_periods[idx_new - 1]
                elif len(available_periods) > 1:
                    old_period = (
                        available_periods[0]
                        if available_periods[0] != new_period
                        else None
                    )
        except Exception:
            pass

    if isinstance(old_period, str):
        old_period = old_period.strip()
    else:
        old_period = None
    if isinstance(new_period, str):
        new_period = new_period.strip()
    else:
        new_period = None

    # Helper: load response text by period, question number and language (with fallback)
    def load_response_text(period_name: str, qn: int, lang: str = "PL") -> str:
        """
        Tries to load response text from:
           RESPONSES_BASE / <lang> / <period_name> / q<qn>_response.txt
        If not found, tries a case-insensitive match in that folder.
        If still not found, tries the other language as fallback.
        Returns the file contents or None if not found.
        """
        if not period_name or qn is None:
            return None

        fname = f"q{qn}_response.txt"
        lang = (lang or "PL").upper()

        def try_read(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return fh.read()
            except Exception:
                return None

        # 1) try using pre-computed RESPONSES_DIR if available (preferred)
        if RESPONSES_DIR:
            path_direct = os.path.join(RESPONSES_DIR, period_name, fname)
            if os.path.exists(path_direct):
                txt = try_read(path_direct)
                if txt is not None:
                    return txt

        # 2) case-insensitive filename search within language folder
        folder = os.path.join(RESPONSES_BASE, lang, period_name)
        if os.path.isdir(folder):
            for fn in os.listdir(folder):
                if fn.lower() == fname.lower():
                    txt = try_read(os.path.join(folder, fn))
                    if txt is not None:
                        return txt

        # 3) fallback to other language (if different)
        other_lang = "EN" if lang == "PL" else "PL"
        path2 = os.path.join(RESPONSES_BASE, other_lang, period_name, fname)
        if os.path.exists(path2):
            txt = try_read(path2)
            if txt is not None:
                return txt

        # 4) case-insensitive filename search within other language folder
        folder2 = os.path.join(RESPONSES_BASE, other_lang, period_name)
        if os.path.isdir(folder2):
            for fn in os.listdir(folder2):
                if fn.lower() == fname.lower():
                    txt = try_read(os.path.join(folder2, fn))
                    if txt is not None:
                        return txt

        return None

    # use currently selected language (from session_state)
    current_lang = st.session_state.get("responses_lang", "PL")
    old_text = (
        load_response_text(old_period, q_number, lang=current_lang)
        if old_period
        else None
    )
    new_text = (
        load_response_text(new_period, q_number, lang=current_lang)
        if new_period
        else None
    )

    # Collect evidence phrases from JSON 'evidence' block (if present)
    evidence_block = data.get("evidence", {}) if isinstance(data, dict) else {}
    phrases = collect_evidence_phrases(evidence_block)
    phrases = [p for p in phrases if len(p.strip()) >= 4]
    phrases = list(dict.fromkeys(phrases))

    # Render two large columns
    left_col, right_col = st.columns([1, 1])
    # Left: Old response (label with period)
    with left_col:
        header = (
            f"Old response ({escape(old_period)})"
            if old_period
            else f"Old response (not available)"
        )
        if old_text:
            highlighted_html = highlight_text(old_text, phrases)
            # card with equal height & scrollable content
            card_html = f"""
            <div class="placeholder-card" tabindex="0">
              <div class="response-card-header">{header}</div>
              <div style="height: 545px;" class="placeholder-body response-card-body">
                <div class="response-card-content">{highlighted_html}</div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        else:
            note = (
                f"No response text found for question Q{q_number} in period: {escape(str(old_period))}"
                if old_period
                else "No old-period response available."
            )
            card_html = f"""
            <div class="placeholder-card" tabindex="0">
              <div class="response-card-header">{header}</div>
              <div style="height: 545px;" class="placeholder-body response-card-body">
                <div class="response-card-content" style="display:flex; align-items:center; justify-content:center; color:#666;">
                  {note}
                </div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    # Right: New response (selected period)
    with right_col:
        header = (
            f"New response ({escape(new_period)})"
            if new_period
            else f"New response (not available)"
        )
        if new_text:
            highlighted_html = highlight_text(new_text, phrases)
            card_html = f"""
            <div class="placeholder-card" tabindex="0">
              <div class="response-card-header">{header}</div>
              <div style="height: 545px;" class="placeholder-body response-card-body">
                <div class="response-card-content">{highlighted_html}</div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        else:
            note = (
                f"No response text found for question Q{q_number} in period: {escape(str(new_period))}"
                if new_period
                else "No new-period response available."
            )
            card_html = f"""
            <div class="placeholder-card" tabindex="0">
            <div class="response-card-header">{header}</div>
              <div style="height: 545px;" class="placeholder-body response-card-body">
                <div class="response-card-content" style="display:flex; align-items:center; justify-content:center; color:#666;">
                  {note}
                </div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Show explanatory note only for Semantic Metric (do not show for Scalar Metric)
    if metric_kind == "Semantic Metric":
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='height:12px'><p>The text <mark>highlighted in yellow</mark> represents evidences recognized by LLM as a judge by evaluating given responses based on actual criterion.</p></div>",
            unsafe_allow_html=True,
        )
# ---------------- end new feature ----------------

# ---------------- New section: Sources & Materials (Old / New) ----------------
# Shows st.expander() for "Old" and "New" sources loaded from
# JSONs/<Country>/<Period>/Sources and Materials/{Old,New}/ matching the selected question.


def _find_matching_file_for_label(paths, display_label):
    """Return the first file in paths that matches the display_label (e.g. 'Q6')."""
    if not display_label or not paths:
        return None
    # match by normalized `extract_q_label(short_label(p))`
    for p in paths:
        try:
            if extract_q_label(short_label(p)) == display_label:
                return p
        except Exception:
            continue
    # fallback: numeric match (first integer in display_label)
    m = re.search(r"(\d{1,4})", str(display_label))
    n = int(m.group(1)) if m else None
    if n is not None:
        for p in paths:
            m2 = re.search(r"(\d{1,4})", os.path.basename(p))
            if m2 and int(m2.group(1)) == n:
                return p
    return None


# Determine the Sources & Materials folder path(s)
sources_folder_name = "Sources and Materials"
sources_paths = []
for base in BASE_DIRS:
    p = os.path.join(base, country, period, sources_folder_name)
    if os.path.isdir(p):
        sources_paths.append(p)

if not sources_paths:
    # No folder -> nothing to show
    pass
else:
    # Collect Old / New files across base dirs (preserve dedup & order)
    side_files = {"Old": [], "New": []}
    for sp in sources_paths:
        for side in ("Old", "New"):
            side_dir = os.path.join(sp, side)
            if os.path.isdir(side_dir):
                found = sorted(glob(os.path.join(side_dir, "*.json")))
                side_files[side].extend(found)

    # Deduplicate & sort
    for side in ("Old", "New"):
        side_files[side] = sorted(list(dict.fromkeys(side_files[side])))

    # Determine the currently selected label, typically "Qn"
    # Prefer the authoritative selectbox value (persisted in session_state) so the
    # question selection remains stable across topic/metric changes.
    selected_label_for_q = st.session_state.get("file_selectbox")
    if not selected_label_for_q:
        # fallback to index-based label (keeps backward-compatibility)
        selected_label_for_q = display_labels[file_index] if display_labels else None

    # Find actual file path for the selected question (if present)
    old_sources_path = _find_matching_file_for_label(
        side_files["Old"], selected_label_for_q
    )
    new_sources_path = _find_matching_file_for_label(
        side_files["New"], selected_label_for_q
    )

    # Safe loader helper for JSONs with "sources_materials" list
    def _load_sources_list(path):
        if not path or not os.path.exists(path):
            return None
        try:
            payload = load_json(path)
            if not isinstance(payload, dict):
                return None
            lst = payload.get("sources_materials")
            if isinstance(lst, list):
                return [
                    str(x).strip() for x in lst if (x is not None and str(x).strip())
                ]
            else:
                return []
        except Exception:
            return None

    old_sources = _load_sources_list(old_sources_path)
    new_sources = _load_sources_list(new_sources_path)

    # ------------------ source comparison helpers ------------------
    def _normalize_text(s: str) -> str:
        """Normalize a citation string for robust comparison."""
        if not s:
            return ""
        # Unicode normalization
        s = unicodedata.normalize("NFKC", str(s))
        # Lowercase, strip
        s = s.lower().strip()
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s)
        # Remove leading bullets and common list markers
        s = re.sub(r"^[\u2022\-\*\•\.\)\s]+", "", s)
        # Remove trailing punctuation (but keep URLs intact; we'll handle URLs separately)
        s = s.rstrip(" .,;:-")
        return s

    def _extract_first_url(s: str):
        """Return a normalized canonical URL string if present, otherwise None."""
        if not s:
            return None
        # crude url find — works for typical citations (http(s)://...)
        m = re.search(r"(https?://[^\s\)\]\}]+)", s, flags=re.IGNORECASE)
        if not m:
            return None
        raw = m.group(1).rstrip(".,;:")
        try:
            p = urlparse(raw)
            netloc = p.netloc.lower()
            # strip www.
            if netloc.startswith("www."):
                netloc = netloc[4:]
            # reconstruct canonical-ish path (no query params)
            path = p.path.rstrip("/")
            canon = f"{netloc}{path}"
            return canon
        except Exception:
            return raw.lower().rstrip("/")

    def _is_match(
        new_s: str, old_s: str, fuzzy_threshold: float = 0.86
    ) -> (bool, float):
        """
        Return (match_bool, score).
        Matching strategy:
          1) if both have URLs and canonical URLs match -> match (score 1.0).
          2) if normalized strings equal -> match (score 1.0).
          3) else compute difflib ratio; if >= threshold, treat as match.
        """
        if not new_s or not old_s:
            return (False, 0.0)
        # Try URL match
        new_url = _extract_first_url(new_s)
        old_url = _extract_first_url(old_s)
        if new_url and old_url:
            if new_url == old_url:
                return (True, 1.0)
        # Exact normalized string
        nn = _normalize_text(new_s)
        on = _normalize_text(old_s)
        if not nn or not on:
            return (False, 0.0)
        if nn == on:
            return (True, 1.0)
        # Fuzzy similarity fallback (difflib)
        # Use SequenceMatcher quick ratio
        score = difflib.SequenceMatcher(a=nn, b=on).ratio()
        if score >= fuzzy_threshold:
            return (True, score)
        return (False, score)

    # Prepare comparison results (map each new item -> (is_new, best_match, best_score))
    new_flags = (
        []
    )  # list of tuples (new_src, is_new_bool, best_match_str_or_None, score_float)
    _old_list = old_sources or []
    _new_list = new_sources or []

    # Defensive: if no old list, mark everything in new as new
    if not _old_list:
        for s in _new_list:
            new_flags.append((s, True, None, 0.0))
    else:
        # For each new source, check best match among old sources
        for ns in _new_list:
            best_score = -1.0
            best_match = None
            matched = False
            for osrc in _old_list:
                is_m, score = _is_match(ns, osrc, fuzzy_threshold=0.86)
                if score > best_score:
                    best_score = score
                    best_match = osrc
                if is_m:
                    matched = True
                    # we still keep best_score/best_match for tooltip info
                    break
            new_flags.append(
                (ns, not matched, best_match if matched else best_match, best_score)
            )
    # ------------------ END: source comparison helpers ---------------

    # Build display columns
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    cols = st.columns([1, 1])
    left_col, right_col = cols[0], cols[1]

    def _sanitized_key_id(s: str) -> str:
        """Return short stable id (md5) for strings/paths; safe for session keys."""
        if not s:
            return "none"
        try:
            # prefer absolute path if it looks like a filesystem path
            if os.path.exists(s):
                key_src = os.path.abspath(s)
            else:
                key_src = str(s)
            return hashlib.md5(key_src.encode("utf-8")).hexdigest()[:12]
        except Exception:
            return re.sub(r"[^0-9A-Za-z]", "_", str(s))[:20]

    # Helper that returns the unique checkbox key to persist expander state for a given side
    def _sources_expander_checkbox_key(side: str, source_path: str, display_label: str):
        """
        Compose a stable key using side ('Old'|'New') and a short id derived from the actual
        matching sources JSON path. If no path exists, fallback to the display_label.
        """
        if source_path:
            idpart = _sanitized_key_id(source_path)
        else:
            idpart = _sanitized_key_id(display_label or "none")
        return f"sources_materials_{side}_{idpart}_cb"

    # For both sides, show a simple expander containing the sources (no visible checkboxes)
    sides_info = [
        ("Old", left_col, old_sources_path, old_sources),
        ("New", right_col, new_sources_path, new_sources),
    ]

    # ----------------- helper: linkify a plain citation string into safe HTML -----------------
    def _linkify_citation(text: str) -> str:
        """
        Find http(s)://... occurrences in `text` and replace each with a safe HTML anchor.
        Non-URL parts are HTML-escaped. Returns HTML string (safe for unsafe_allow_html=True).
        """
        if not text:
            return ""

        parts = []
        last_end = 0
        # regex: find http/https urls up to whitespace or closing parens/brackets; we'll strip trailing punctuation
        url_re = re.compile(r"https?://[^\s\)\]\}]+", flags=re.IGNORECASE)
        for m in url_re.finditer(text):
            start, end = m.start(), m.end()
            # append escaped text before URL
            if start > last_end:
                parts.append(escape(text[last_end:start]))
            raw_url = m.group(0).rstrip(
                ".,;:"
            )  # remove trailing punctuation if present
            safe_url = escape(raw_url)
            # create anchor that opens in new tab (and is safe)
            anchor = f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{safe_url}</a>'
            parts.append(anchor)
            last_end = end
        # append remainder
        if last_end < len(text):
            parts.append(escape(text[last_end:]))

        return "".join(parts)

    # ----------------- Render Old / New expanders (Old = plain, New = flagged + linkified) -----------------
    for side, col_obj, src_path, src_list in sides_info:
        with col_obj:
            title = f"Sources & Materials — {side} response"

            # Normal expander — let Streamlit remember open/closed state automatically.
            with st.expander(title):
                if src_list is None:
                    st.info(f"No sources file found for this question ({side}).")
                    continue
                if not src_list:
                    st.write("_No sources listed for this response._")
                    continue

                # OLD: render as simple markdown list but linkify URLs so links are clickable too
                if side == "Old":
                    for src in src_list:
                        # convert any URL to clickable anchor, keep other text escaped
                        linked_html = _linkify_citation(src)
                        # if no URL was present, _linkify_citation returns escaped text, but we want a markdown bullet.
                        # Use a small HTML fragment to match earlier style (use dash + space)
                        st.markdown(f"- {linked_html}", unsafe_allow_html=True)

                # NEW: render using new_flags — build a consistent <ul><li> list so spacing matches Old side
                else:  # side == "New"
                    # Build an HTML unordered list so spacing/appearance matches the Old side's markdown list items
                    # Choose margin-bottom that visually matches Old; adjust px if you want slightly different spacing
                    LI_MARGIN_PX = (
                        17  # tweak this so spacing equals Old side (10 is typical)
                    )
                    ul_items = []
                    ul_items.append(
                        f'<ul style="padding-left:18px; margin-top:0; margin-bottom:0;">'
                    )
                    for src, is_new, best_match, score in new_flags:
                        linked_html = _linkify_citation(src)
                        if is_new:
                            if best_match:
                                tooltip = f"closest match: {escape(best_match)} — score={score:.2f}"
                            else:
                                tooltip = "no similar source in Old list"
                            # Put icon inside <span> with title for tooltip; keep same text/link structure as others
                            li_html = (
                                f'<li style="margin-bottom:{LI_MARGIN_PX}px;">'
                                f'<span style="font-weight:700; color:#c0392b; margin-right:6px;" title="{escape(tooltip)}">🔔</span>'
                                f"{linked_html}"
                                f"</li>"
                            )
                        else:
                            li_html = f'<li style="margin-bottom:{LI_MARGIN_PX}px;">{linked_html}</li>'
                        ul_items.append(li_html)
                    ul_items.append("</ul>")
                    full_ul_html = "".join(ul_items)
                    # render the whole list at once (safe: linkify already escaped non-URL text)
                    st.markdown(full_ul_html, unsafe_allow_html=True)
