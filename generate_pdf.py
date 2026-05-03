"""
generate_pdf.py — FIFTO Intraday Option Selling System — Client PDF
====================================================================
Generates: FIFTO_Intraday_Selling_System.pdf
"""
import os
import io
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, Circle as MplCircle, Arc, Wedge as MplWedge
from matplotlib.lines import Line2D
from matplotlib import patheffects as mpe

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether, Image as RLImage,
    FrameBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Circle, Rect, Polygon, String, Line as RLLine
from reportlab.graphics import renderPDF

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__),
                        "data", "20260503", "127_all_trades.csv")
OUT_PATH = "FIFTO_Intraday_Selling_System.pdf"

# ── Color Palette ────────────────────────────────────────────────────────────
DARK        = colors.HexColor("#0D1117")
DARK2       = colors.HexColor("#161B22")
GOLD        = colors.HexColor("#F0B90B")
GOLD_LIGHT  = colors.HexColor("#FDD835")
BLUE        = colors.HexColor("#1565C0")
LIGHT_BLUE  = colors.HexColor("#1E3A5F")
NAVY        = colors.HexColor("#0D2137")
RED         = colors.HexColor("#C62828")
GREEN       = colors.HexColor("#2E7D32")
SILVER      = colors.HexColor("#607D8B")
WHITE       = colors.white
LIGHT_GREY  = colors.HexColor("#F5F5F5")
MID_GREY    = colors.HexColor("#ECEFF1")
DARK_GREY   = colors.HexColor("#263238")
ACCENT      = colors.HexColor("#FF6F00")
GREEN_LIGHT = colors.HexColor("#E8F5E9")
RED_LIGHT   = colors.HexColor("#FFEBEE")

THOR_C   = "#1565C0"
HULK_C   = "#2E7D32"
IRON_C   = "#C62828"
CAP_C    = "#0D47A1"
SPIDER_C = "#B71C1C"
WIDOW_C  = "#37474F"
HAWK_C   = "#6A1B9A"

# ── Page size ────────────────────────────────────────────────────────────────
W, H = A4   # 595 x 842 pts

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — STYLES
# ─────────────────────────────────────────────────────────────────────────────
def S(name, **kw): return ParagraphStyle(name, **kw)

COVER_TITLE = S("ct", fontSize=46, textColor=GOLD, alignment=TA_CENTER,
                fontName="Helvetica-Bold", leading=54, spaceAfter=6)
COVER_SUB   = S("cs", fontSize=18, textColor=WHITE, alignment=TA_CENTER,
                fontName="Helvetica", leading=26, spaceAfter=4)
COVER_TAG   = S("ctag", fontSize=11, textColor=SILVER, alignment=TA_CENTER,
                fontName="Helvetica-Oblique", leading=17)
COVER_STAT_LABEL = S("csl", fontSize=9, textColor=SILVER, alignment=TA_CENTER,
                     fontName="Helvetica-Bold", leading=13)
COVER_STAT_VAL   = S("csv", fontSize=14, textColor=GOLD, alignment=TA_CENTER,
                     fontName="Helvetica-Bold", leading=18)

H1 = S("h1", fontSize=22, textColor=GOLD, fontName="Helvetica-Bold",
        leading=28, spaceBefore=12, spaceAfter=6)
H2 = S("h2", fontSize=15, textColor=LIGHT_BLUE, fontName="Helvetica-Bold",
        leading=20, spaceBefore=10, spaceAfter=4)
H3 = S("h3", fontSize=12, textColor=DARK_GREY, fontName="Helvetica-Bold",
        leading=16, spaceBefore=6, spaceAfter=3)
BODY   = S("body", fontSize=10, textColor=DARK_GREY, fontName="Helvetica",
           leading=15, spaceBefore=3, spaceAfter=3, alignment=TA_JUSTIFY)
SMALL  = S("small", fontSize=8.5, textColor=SILVER, fontName="Helvetica",
           leading=12, spaceBefore=2)
RULE_B = S("rule", fontSize=10, textColor=DARK_GREY, fontName="Helvetica",
           leading=16, spaceBefore=2, spaceAfter=2, leftIndent=12)
AGENT_T = S("at", fontSize=14, textColor=WHITE, fontName="Helvetica-Bold",
             leading=18, alignment=TA_LEFT)
AGENT_S = S("asub", fontSize=10, textColor=MID_GREY, fontName="Helvetica-Oblique",
             leading=14, alignment=TA_LEFT)

def agent_name_style(color):
    return S("an", fontSize=18, textColor=color, fontName="Helvetica-Bold",
              leading=24, spaceBefore=6, spaceAfter=4)

def hr(): return HRFlowable(width="100%", thickness=1.5,
                             color=GOLD, spaceAfter=8, spaceBefore=6)
def hr_thin(): return HRFlowable(width="100%", thickness=0.5,
                                  color=colors.HexColor("#CFD8DC"),
                                  spaceAfter=6, spaceBefore=6)

def tblstyle(hdr=LIGHT_BLUE, fs=9):
    return TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), hdr),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), fs),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,1), (-1,-1), fs - 1),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LIGHT_GREY]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#B0BEC5")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_stats():
    df = pd.read_csv(CSV_PATH)
    df["date"] = df["date"].astype(str)

    def map_agent(row):
        s = row["signal"]
        m = {"v17a":"THOR","cam_l3":"HULK","cam_h3":"IRON MAN",
             "iv2_r1":"IRON MAN","iv2_r2":"IRON MAN","iv2_pdl":"CAPTAIN",
             "CRT":"SPIDER-MAN","MRC":"BLACK WIDOW","S4_2nd":"HAWKEYE"}
        return m.get(s, "OTHER")

    df["agent"] = df.apply(map_agent, axis=1)
    df["ym"]    = df["date"].str[:6]
    df["yr"]    = df["date"].str[:4].astype(int)

    agent_g = df.groupby("agent").agg(
        trades   = ("pnl", "count"),
        wr       = ("win", "mean"),
        total    = ("pnl", "sum"),
        avg      = ("pnl", "mean"),
        hard_sl  = ("exit_reason", lambda x: (x=="hard_sl").sum()),
        target_n = ("exit_reason", lambda x: (x=="target").sum()),
    ).round(2)
    agent_g["wr_pct"]       = (agent_g["wr"] * 100).round(1)
    agent_g["hard_sl_pct"]  = (agent_g["hard_sl"] / agent_g["trades"] * 100).round(1)
    agent_g["target_pct"]   = (agent_g["target_n"] / agent_g["trades"] * 100).round(1)
    agent_g["total_fmt"]    = agent_g["total"].apply(lambda x: f"Rs.{x:,.0f}")
    agent_g["avg_fmt"]      = agent_g["avg"].apply(lambda x: f"Rs.{x:,.0f}")

    yr_g = df.groupby("yr").agg(
        trades = ("pnl", "count"),
        wr     = ("win", "mean"),
        total  = ("pnl", "sum"),
        avg    = ("pnl", "mean"),
    ).round(2)

    monthly = df.groupby("ym")["pnl"].sum().reset_index()
    monthly.columns = ["ym", "pnl"]

    peak  = df["cum_pnl"].cummax()
    dd    = df["cum_pnl"] - peak
    max_dd = abs(dd.min())

    return df, agent_g, yr_g, monthly, max_dd

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CHART GENERATION
# ─────────────────────────────────────────────────────────────────────────────

MPL_STYLE = {
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#C9D1D9",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "text.color":       "#C9D1D9",
    "grid.color":       "#21262D",
    "grid.linewidth":   0.5,
    "axes.spines.top":  False,
    "axes.spines.right":False,
}


def make_equity_chart(df):
    """Equity curve + drawdown, dark theme."""
    with plt.rc_context(MPL_STYLE):
        fig = plt.figure(figsize=(12, 7))
        gs  = fig.add_gridspec(3, 1, hspace=0.08)
        ax1 = fig.add_subplot(gs[:2, 0])
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)

        x    = np.arange(len(df))
        eq   = df["cum_pnl"].values / 1e5   # in lakhs
        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak)

        # Equity curve
        ax1.fill_between(x, eq, alpha=0.18, color="#F0B90B")
        ax1.plot(x, eq, color="#F0B90B", linewidth=1.8, label="Cumulative P&L")
        ax1.plot(x, peak, color="#4FC3F7", linewidth=0.8, alpha=0.5,
                 linestyle="--", label="Peak Equity")
        ax1.set_ylabel("₹ Lakhs", fontsize=10, labelpad=6)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"₹{v:.1f}L"))
        ax1.grid(True)
        ax1.legend(loc="upper left", fontsize=9,
                   facecolor="#161B22", edgecolor="#30363D",
                   labelcolor="#C9D1D9")

        # Year dividers
        yr_changes = df[df["date"].str[4:8] == "0101"].index.tolist()
        for idx in yr_changes:
            ax1.axvline(idx, color="#30363D", linewidth=0.8)

        # Year labels
        for yr in [2021, 2022, 2023, 2024, 2025, 2026]:
            mask = df["date"].str[:4] == str(yr)
            idxs = np.where(mask)[0]
            if len(idxs):
                mid = idxs[len(idxs)//2]
                ax1.text(mid, ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else 0.05,
                         str(yr), color="#8B949E", fontsize=8,
                         ha="center", va="bottom")

        ax1.set_title("FIFTO — 5-Year Equity Curve (949 Trades · 2021–2026)",
                      fontsize=13, color="#F0B90B", pad=10, fontweight="bold")

        # Drawdown
        ax2.fill_between(x, dd, 0, color="#F85149", alpha=0.4)
        ax2.plot(x, dd, color="#F85149", linewidth=1)
        ax2.set_ylabel("Drawdown (₹L)", fontsize=9, labelpad=6)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"₹{v:.1f}L"))
        ax2.set_xlabel("Trade #", fontsize=9)
        ax2.grid(True)
        ax2.set_xlim(0, len(df) - 1)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150,
                    facecolor="#0D1117", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
    return buf


def make_monthly_chart(monthly):
    """Monthly P&L bar chart, dark theme."""
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(12, 4.5))

        pnl  = monthly["pnl"].values / 1000    # in thousands
        cols = ["#3FB950" if v >= 0 else "#F85149" for v in pnl]
        x    = np.arange(len(pnl))
        ax.bar(x, pnl, color=cols, width=0.7, zorder=2)
        ax.axhline(0, color="#8B949E", linewidth=0.8)

        # Year dividers
        yrs = monthly["ym"].str[:4]
        prev = None
        for i, y in enumerate(yrs):
            if y != prev:
                if prev is not None:
                    ax.axvline(i - 0.5, color="#30363D", linewidth=1)
                ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -max(abs(pnl))*0.15,
                        y, color="#8B949E", fontsize=8.5, ha="left", va="top")
                prev = y

        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"₹{v:.0f}K"))
        ax.set_xticks([])
        ax.set_title("Monthly P&L — 58 Months (Jan 2021 – Apr 2026)",
                     fontsize=12, color="#F0B90B", pad=10, fontweight="bold")
        ax.grid(axis="y", zorder=1)

        # Avg line
        avg = pnl.mean()
        ax.axhline(avg, color="#F0B90B", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(len(pnl) - 1, avg + abs(pnl).max() * 0.03,
                f"Avg ₹{avg:.0f}K", color="#F0B90B", fontsize=8,
                ha="right")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150,
                    facecolor="#0D1117", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
    return buf


def make_agent_emblem(name, color_hex, letter, accent="circle"):
    """
    Create a circular agent badge PNG.
    accent: 'bolt' | 'star' | 'web' | 'ring' | 'hour' | 'target' | 'fist' | 'circle'
    """
    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=130)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Shadow
    shadow = MplCircle((52, 48), 44, color="#00000025", zorder=1)
    ax.add_patch(shadow)
    # Background
    main = MplCircle((50, 50), 43, color=color_hex, zorder=2)
    ax.add_patch(main)
    # Gold rim
    rim = MplCircle((50, 50), 43, fill=False,
                    edgecolor="#F0B90B", linewidth=2.5, zorder=3)
    ax.add_patch(rim)
    # Inner subtle ring
    inner = MplCircle((50, 50), 37, fill=False,
                      edgecolor=(1, 1, 1, 0.15), linewidth=1, zorder=3)
    ax.add_patch(inner)

    # ── Accent shapes ─────────────────────────────────────────────────────
    if accent == "bolt":            # THOR — lightning bolt
        bolt_x = [46, 54, 49, 57, 43, 51, 46]
        bolt_y = [70, 70, 55, 55, 35, 35, 70]
        ax.fill(bolt_x, bolt_y, color="#F0B90B", alpha=0.9, zorder=3)
        ax.plot(bolt_x, bolt_y, color="#C8860A", linewidth=0.8, zorder=4)

    elif accent == "fist":          # HULK — two horizontal bars
        for dy in [3, -3]:
            rect = FancyBboxPatch((32, 47+dy*3), 36, 5,
                                  boxstyle="round,pad=0.5",
                                  color=(1,1,1,0.2), zorder=3)
            ax.add_patch(rect)

    elif accent == "ring":          # IRON MAN — arc reactor rings
        for r, alpha in [(28, 0.35), (20, 0.25), (10, 0.4)]:
            c = MplCircle((50, 50), r, fill=False,
                          edgecolor="#F0B90B", linewidth=1.5, alpha=alpha, zorder=3)
            ax.add_patch(c)
        # Center dot
        dot = MplCircle((50, 50), 5, color="#F0B90B", alpha=0.8, zorder=4)
        ax.add_patch(dot)

    elif accent == "star":          # CAPTAIN — star
        star_pts = []
        for i in range(10):
            angle = np.pi/2 + i * 2*np.pi/10
            r = 22 if i % 2 == 0 else 10
            star_pts.append([50 + r*np.cos(angle), 50 + r*np.sin(angle)])
        xs, ys = zip(*star_pts)
        ax.fill(xs, ys, color="#F0B90B", alpha=0.35, zorder=3)
        ax.plot(list(xs)+[xs[0]], list(ys)+[ys[0]],
                color="#F0B90B", alpha=0.6, linewidth=1, zorder=4)

    elif accent == "web":           # SPIDER-MAN — web lines
        cx, cy = 50, 50
        for angle in np.linspace(0, np.pi*2, 8, endpoint=False):
            ax.plot([cx, cx + 38*np.cos(angle)],
                    [cy, cy + 38*np.sin(angle)],
                    color=(1,1,1,0.15), linewidth=0.8, zorder=3)
        for r in [10, 20, 30, 38]:
            c = MplCircle((cx, cy), r, fill=False,
                          edgecolor=(1,1,1,0.12), linewidth=0.8, zorder=3)
            ax.add_patch(c)

    elif accent == "hour":          # BLACK WIDOW — hourglass
        tri_t = plt.Polygon([[38,72],[62,72],[50,55]], closed=True,
                             color=(1,1,1,0.3), zorder=3)
        tri_b = plt.Polygon([[38,28],[62,28],[50,45]], closed=True,
                             color=(1,1,1,0.3), zorder=3)
        ax.add_patch(tri_t)
        ax.add_patch(tri_b)

    elif accent == "target":        # HAWKEYE — bullseye + crosshair
        for r, alpha in [(30, 0.15), (20, 0.20), (10, 0.30)]:
            c = MplCircle((50, 50), r, fill=False,
                          edgecolor="#F0B90B", linewidth=1.5, alpha=alpha, zorder=3)
            ax.add_patch(c)
        ax.plot([50, 50], [22, 78], color="#F0B90B", alpha=0.3, linewidth=1, zorder=3)
        ax.plot([22, 78], [50, 50], color="#F0B90B", alpha=0.3, linewidth=1, zorder=3)

    # Large letter
    ax.text(50, 53, letter, fontsize=44, color="white",
            ha="center", va="center", fontweight="bold",
            fontfamily="DejaVu Sans", zorder=5,
            path_effects=[mpe.withStroke(linewidth=2, foreground=color_hex)])

    # Agent name at bottom arc
    ax.text(50, 15, name, fontsize=8.5, color="#F0B90B",
            ha="center", va="center", fontweight="bold",
            fontfamily="DejaVu Sans", zorder=5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                facecolor="white", dpi=130, pad_inches=0.05)
    buf.seek(0)
    plt.close(fig)
    return buf


def make_wr_gauge(wr_pct, color_hex, size=1.8):
    """Mini donut gauge for win rate."""
    fig, ax = plt.subplots(figsize=(size, size), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    cx, cy, r = 0.5, 0.5, 0.42
    # Background ring
    bg = MplWedge((cx, cy), r, 0, 360, width=0.13,
                  facecolor="#ECEFF1", edgecolor="none", zorder=1)
    ax.add_patch(bg)
    # Filled arc
    fill = MplWedge((cx, cy), r, 90 - wr_pct*3.6, 90, width=0.13,
                    facecolor=color_hex, edgecolor="none", zorder=2)
    ax.add_patch(fill)
    # Center text
    ax.text(cx, cy + 0.04, f"{wr_pct:.0f}%", fontsize=14, color="#263238",
            ha="center", va="center", fontweight="bold")
    ax.text(cx, cy - 0.12, "WIN RATE", fontsize=5.5, color="#607D8B",
            ha="center", va="center")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                facecolor="white", dpi=120, pad_inches=0.03)
    buf.seek(0)
    plt.close(fig)
    return buf


def buf_to_rl_image(buf, width_cm, height_cm=None):
    """Convert BytesIO PNG to ReportLab Image flowable."""
    img = RLImage(buf, width=width_cm*cm,
                  height=height_cm*cm if height_cm else None)
    img.hAlign = "CENTER"
    return img


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PAGE CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
def _draw_cover_bg(canvas, doc):
    """Full-page dark background + gold accent bars for cover page."""
    canvas.saveState()
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # top gold bar (full width, above margins)
    canvas.setFillColor(GOLD)
    canvas.rect(0, H - 6, W, 6, fill=1, stroke=0)
    # bottom gold bar
    canvas.rect(0, 0, W, 6, fill=1, stroke=0)
    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — BUILD
# ─────────────────────────────────────────────────────────────────────────────
def build():
    df, ag, yr_g, monthly, max_dd = load_stats()

    # Pre-generate charts once
    equity_buf  = make_equity_chart(df)
    monthly_buf = make_monthly_chart(monthly)

    # Agent emblem configs
    EMBLEMS = [
        ("THOR",        THOR_C,   "T", "bolt"),
        ("HULK",        HULK_C,   "H", "fist"),
        ("IRON MAN",    IRON_C,   "I", "ring"),
        ("CAPTAIN",     CAP_C,    "C", "star"),
        ("SPIDER-MAN",  SPIDER_C, "S", "web"),
        ("BLACK WIDOW", WIDOW_C,  "W", "hour"),
        ("HAWKEYE",     HAWK_C,   "H", "target"),
    ]
    emblem_bufs = {
        name: make_agent_emblem(name, clr, ltr, acc)
        for name, clr, ltr, acc in EMBLEMS
    }
    gauge_bufs = {}
    for name, clr, ltr, acc in EMBLEMS:
        if name in ag.index:
            gauge_bufs[name] = make_wr_gauge(ag.loc[name, "wr_pct"], clr)

    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=A4,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
        leftMargin=2.0*cm, rightMargin=2.0*cm,
    )
    story = []

    # ══════════════════════════════════════════════════════════════
    # PAGE 1 — COVER  (dark background drawn by _draw_cover_bg)
    # ══════════════════════════════════════════════════════════════
    story.append(Spacer(1, 1.6*cm))

    story.append(HRFlowable(width="100%", thickness=2.5, color=GOLD,
                             hAlign="CENTER", spaceAfter=0.7*cm))

    story.append(Paragraph("FIFTO",
        S("cv_t", fontSize=68, textColor=GOLD, alignment=TA_CENTER,
          fontName="Helvetica-Bold", leading=76, spaceAfter=6)))

    story.append(Paragraph("Fusion Intraday Formula for Tactical Options",
        S("cv_s", fontSize=16, textColor=WHITE, alignment=TA_CENTER,
          fontName="Helvetica", leading=22, spaceAfter=0)))

    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph(
        "NIFTY Weekly Options  ·  Intraday Option Selling System",
        S("cv_tag1", fontSize=11, textColor=SILVER, alignment=TA_CENTER,
          fontName="Helvetica-Oblique", leading=16)))

    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph(
        "5-Year Verified Performance  ·  2021–2026  ·  949 Trades",
        S("cv_tag2", fontSize=11, textColor=SILVER, alignment=TA_CENTER,
          fontName="Helvetica-Oblique", leading=16)))

    story.append(Spacer(1, 0.7*cm))

    story.append(HRFlowable(width="65%", thickness=1, color=GOLD,
                             hAlign="CENTER", spaceAfter=0.25*cm))
    story.append(Paragraph("— The Avengers of the Market —",
        S("cv_av", fontSize=13, textColor=GOLD, alignment=TA_CENTER,
          fontName="Helvetica-Bold", leading=18)))
    story.append(HRFlowable(width="65%", thickness=1, color=GOLD,
                             hAlign="CENTER", spaceAfter=0.8*cm))

    # 7 agent emblems in a row
    _emb_cw = 16.2 * cm / 7
    _cv_emb_row = [buf_to_rl_image(emblem_bufs[n], 1.9, 1.9)
                   for n, _, _, _ in EMBLEMS]
    _cv_name_row = [
        Paragraph(f"<font color='{c}'><b>{n}</b></font>",
                  S(f"cvn{i}", fontSize=7, fontName="Helvetica-Bold",
                    textColor=colors.HexColor(c), alignment=TA_CENTER, leading=9))
        for i, (n, c, _, _) in enumerate(EMBLEMS)
    ]
    _cv_emb_tbl = Table(
        [_cv_emb_row, _cv_name_row],
        colWidths=[_emb_cw] * 7,
        rowHeights=[2.05*cm, 0.45*cm])
    _cv_emb_tbl.setStyle(TableStyle([
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    story.append(_cv_emb_tbl)
    story.append(Spacer(1, 0.9*cm))

    # Key stats row
    _cvl = S("cvl", fontSize=9,  textColor=SILVER, alignment=TA_CENTER,
              fontName="Helvetica-Bold", leading=12)
    _cvv = S("cvv", fontSize=18, textColor=GOLD,   alignment=TA_CENTER,
              fontName="Helvetica-Bold", leading=22)
    _cv_stats = Table(
        [[Paragraph("TOTAL P&amp;L", _cvl),
          Paragraph("WIN RATE",       _cvl),
          Paragraph("MAX DRAWDOWN",   _cvl),
          Paragraph("COVERAGE",       _cvl)],
         [Paragraph("Rs.16,96,299",   _cvv),
          Paragraph("74.5%",          _cvv),
          Paragraph("2.93%",          _cvv),
          Paragraph("65.2%",          _cvv)]],
        colWidths=[4.5*cm, 3.5*cm, 4.5*cm, 3.7*cm],
        rowHeights=[0.75*cm, 1.35*cm])
    _cv_stats.setStyle(TableStyle([
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("LINEBELOW",     (0,0), (-1,0), 0.5, colors.HexColor("#444444")),
        ("LINEAFTER",     (0,0), (2,-1), 0.5, colors.HexColor("#444444")),
        ("BOX",           (0,0), (-1,-1), 1,   colors.HexColor("#555555")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(_cv_stats)
    story.append(Spacer(1, 0.7*cm))

    story.append(Paragraph(
        "3 of 58 months negative  ·  Zero overnight holding  ·  Fully mechanical execution",
        S("cv_3neg", fontSize=10, textColor=SILVER, alignment=TA_CENTER,
          fontName="Helvetica-Oblique", leading=15)))

    story.append(Spacer(1, 0.6*cm))

    story.append(Paragraph("7 Agents  ·  7 Zones  ·  One Systematic Framework",
        S("cv_7a", fontSize=11, textColor=SILVER, alignment=TA_CENTER,
          fontName="Helvetica", leading=16)))

    story.append(Spacer(1, 3.2*cm))

    story.append(HRFlowable(width="100%", thickness=2.5, color=GOLD,
                             hAlign="CENTER", spaceAfter=0.4*cm))

    story.append(Paragraph("Confidential  ·  For Authorized Recipients Only",
        S("cv_conf", fontSize=9, textColor=SILVER, alignment=TA_CENTER,
          fontName="Helvetica", leading=13)))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # PAGE 2 — WHAT IS FIFTO
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("What is FIFTO?", H1))
    story.append(hr())
    story.append(Paragraph(
        "FIFTO is a <b>rules-based intraday option selling system</b> built on NIFTY weekly "
        "options. It identifies high-probability price zones where the market is likely to "
        "reverse or stall, and sells option premium at those zones — capturing time decay and "
        "mean reversion with a disciplined, fully mechanical exit framework.", BODY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Seven independent strategies — each named after a Marvel Avenger — operate across "
        "different price zones. Every agent has precise zone-based triggers, a fixed 30% "
        "profit target, and an automated trailing stop-loss. <b>Human judgement is removed "
        "from the execution loop.</b>", BODY))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Five Core Principles", H2))
    principles = [
        ("Zone-First", "Never sell blindly. Every trade requires spot price to be at a defined technical zone computed from previous-day data. No discretionary entries."),
        ("Premium Sell Only", "All trades are option SELL (short premium). We collect theta decay, not pay it. CE sell on bearish setups. PE sell on bullish reversals."),
        ("Mechanical Exit", "Target = 30% of entry premium. Trailing SL activates automatically at defined milestones. No manual intervention once trade is active."),
        ("One Trade Per Day", "Maximum one active position at a time. No averaging, no doubling down, no revenge trading. Capital is protected by design."),
        ("Capital Safety First", "Hard SL = 100% of entry premium. Negative months in 5 years: only 3 out of 58. Maximum drawdown: 2.93% of peak equity."),
    ]
    for i, (title, desc) in enumerate(principles, 1):
        story.append(Paragraph(f"<b>{i}. {title}:</b> {desc}", RULE_B))
        story.append(Spacer(1, 0.1*cm))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("System at a Glance", H2))
    glance_data = [
        ["Metric",                "Value",         "Metric",                "Value"],
        ["Total Trades",          "949",           "Trading Instrument",    "NIFTY Weekly Options"],
        ["Win Rate",              "74.5%",         "Trade Type",            "Option Sell (Short Premium)"],
        ["Total PnL (5yr)",       "Rs.16,96,299",  "Lot Size",              "65 shares / lot"],
        ["Avg Monthly PnL",       "Rs.29,247",     "Max Lots per Trade",    "3 lots"],
        ["Max Drawdown",          "2.93%",         "Hard SL Trigger",       "Premium doubles (100% loss)"],
        ["Negative Months",       "3 / 58",        "Target per Trade",      "30% of entry premium"],
        ["Best Month",            "Rs.1,08,030",   "Entry Window",          "09:16 to 15:15"],
        ["Return on Capital",     "~33.9% p.a.",   "EOD Exit",              "15:20:00 — no overnight"],
    ]
    glance_tbl = Table(glance_data,
                       colWidths=[3.8*cm, 3.2*cm, 3.8*cm, 5.4*cm])
    g_ts = TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), DARK),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LIGHT_GREY]),
        ("BACKGROUND",    (2,1), (2,-1), MID_GREY),
        ("FONTNAME",      (2,1), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (2,1), (2,-1), 8.5),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#B0BEC5")),
        ("ALIGN",         (0,0), (-1,-1), "LEFT"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ])
    glance_tbl.setStyle(g_ts)
    story.append(glance_tbl)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # PAGE 3 — EQUITY CURVE
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Performance Equity Curve — 2021 to 2026", H1))
    story.append(hr())

    # Top KPI row
    kpis = [
        ["TOTAL P&L",      "Rs. 16,96,299"],
        ["WIN RATE",       "74.5%"],
        ["TRADES",         "949"],
        ["MONTHS",         "58 (3 negative)"],
        ["MAX DRAWDOWN",   "Rs. 46,134 (2.93%)"],
        ["AVG/MONTH",      "Rs. 29,247"],
    ]
    kpi_tbl = Table(
        [[Paragraph(f"<b>{k}</b>",
                    S("kl", fontSize=8, textColor=SILVER, fontName="Helvetica-Bold",
                      alignment=TA_CENTER, leading=11)),
          Paragraph(v,
                    S("kv", fontSize=14, textColor=GOLD, fontName="Helvetica-Bold",
                      alignment=TA_CENTER, leading=18))]
         for k, v in kpis],
        colWidths=[8*cm, 8*cm], rowHeights=[1.3*cm]*6)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), DARK),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("GRID",          (0,0), (-1,-1), 1, GOLD),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 0.4*cm))

    story.append(buf_to_rl_image(equity_buf, 16, 9.5))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Equity grows consistently from 2021 to 2026. Maximum drawdown of Rs. 46,134 occurred "
        "July 2024 and recovered within 6 weeks. Zero back-to-back loss months in 5 years.", SMALL))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # PAGE 4 — SEVEN AGENTS OVERVIEW
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("The Seven Agents — Overview", H1))
    story.append(hr())
    story.append(Paragraph(
        "FIFTO deploys seven specialized agents — each targeting a distinct price zone. "
        "Base agents (THOR, HULK, IRON MAN, CAPTAIN) trade on <b>base days</b>. "
        "SPIDER-MAN and BLACK WIDOW activate on <b>blank days</b> when no base signal fires. "
        "HAWKEYE re-enters after any base agent hits target before 13:30.", BODY))
    story.append(Spacer(1, 0.3*cm))

    # Agent overview grid with emblems
    overview_rows = [["", "Agent", "Zone", "Opt", "Lots", "WR", "5yr P&L", "Type"]]
    agent_configs = [
        ("THOR",        THOR_C,   "TC–PDH / R1–R2",       "PE",   "1–3", "74.1%", "Rs.8,56,859", "Base"),
        ("HULK",        HULK_C,   "PDH–R1 (CAM L3)",      "PE",   "1–3", "88.5%", "Rs.1,33,188", "Base"),
        ("IRON MAN",    IRON_C,   "R1 / R2 / CAM H3",     "PE/CE","1–3", "73.0%", "Rs.1,43,107", "Base"),
        ("CAPTAIN",     CAP_C,    "PDL / R1 / R2 (IV2)",  "CE/PE","1",   "60.9%", "Rs.18,561",   "Base"),
        ("SPIDER-MAN",  SPIDER_C, "TC / R1 Sweep Trap",   "CE",   "1",   "69.9%", "Rs.74,935",   "Blank"),
        ("BLACK WIDOW", WIDOW_C,  "l_382 Fibonacci",      "PE",   "2",   "80.6%", "Rs.3,07,346", "Blank"),
        ("HAWKEYE",     HAWK_C,   "Post-target Re-entry", "Same", "1",   "71.4%", "Rs.1,62,302", "Re-entry"),
    ]

    for name, clr, zone, opt, lots, wr, pnl, typ in agent_configs:
        emb = buf_to_rl_image(emblem_bufs[name], 1.0, 1.0)
        wr_color = GREEN if float(wr.replace("%","")) >= 75 else \
                   BLUE  if float(wr.replace("%","")) >= 70 else RED
        overview_rows.append([
            emb,
            Paragraph(f"<b><font color='{clr}'>{name}</font></b>",
                      S("on", fontSize=9, fontName="Helvetica-Bold",
                        textColor=colors.HexColor(clr), leading=12)),
            zone, opt, lots,
            Paragraph(f"<b>{wr}</b>",
                      S("wr", fontSize=9, fontName="Helvetica-Bold",
                        textColor=wr_color, leading=12, alignment=TA_CENTER)),
            pnl, typ,
        ])

    ov_tbl = Table(overview_rows,
                   colWidths=[1.2*cm, 3.0*cm, 3.8*cm, 1.5*cm, 1.2*cm, 1.5*cm, 3.0*cm, 2.0*cm])
    ov_ts = tblstyle(DARK)
    ov_ts.add("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT_GREY])
    ov_ts.add("ALIGN", (0,0), (-1,-1), "CENTER")
    ov_ts.add("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ov_ts.add("TOPPADDING",    (0,1), (-1,-1), 4)
    ov_ts.add("BOTTOMPADDING", (0,1), (-1,-1), 4)
    ov_tbl.setStyle(ov_ts)
    story.append(ov_tbl)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Base Days: THOR/HULK/IRON MAN/CAPTAIN active + HAWKEYE (if target before 13:30). "
        "Blank Days: SPIDER-MAN and BLACK WIDOW active. "
        "Coverage: 65.2% of 1,155 trading days.", SMALL))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Technical Level Computation Reference", H2))
    story.append(Paragraph(
        "All levels are calculated from <b>previous day OHLC</b> before market open at 08:55 AM. "
        "No intraday or future data is used.", BODY))
    story.append(Spacer(1, 0.2*cm))
    level_ref = [
        ["Variable",       "Formula",                          "Used By"],
        ["Pivot (PP)",     "(H + L + C) / 3",                 "THOR, IRON MAN, CAPTAIN"],
        ["BC",             "(H + L) / 2",                     "THOR"],
        ["TC",             "2 × PP - BC",                     "THOR, SPIDER-MAN"],
        ["R1",             "2 × PP - L",                      "IRON MAN, CAPTAIN, SPIDER-MAN"],
        ["R2",             "PP + (H - L)",                    "IRON MAN, CAPTAIN"],
        ["CAM L3",         "Close - Range × 0.275",           "HULK"],
        ["CAM H3",         "Close + Range × 0.275",           "IRON MAN"],
        ["MRC l_382",      "PDH - Range × 0.382",             "BLACK WIDOW (entry zone)"],
        ["PDH / PDL",      "Previous Day High / Low",         "THOR, BLACK WIDOW / CAPTAIN"],
        ["EMA (20-day)",   "Exponential MA of daily close",   "THOR (directional bias filter)"],
        ["Futures Basis",  "Futures LTP - Spot LTP at 09:15","SPIDER-MAN, THOR, HULK basis filter"],
        ["IB High/Low",    "Max/Min spot 09:15–09:45",        "SPIDER-MAN (IB expansion filter)"],
    ]
    ref_tbl = Table(level_ref, colWidths=[2.8*cm, 5.6*cm, 7.8*cm])
    ref_ts = tblstyle(DARK, fs=8)
    ref_ts.add("TOPPADDING",    (0,0), (-1,-1), 2)
    ref_ts.add("BOTTOMPADDING", (0,0), (-1,-1), 2)
    ref_tbl.setStyle(ref_ts)
    story.append(ref_tbl)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # PAGES 5–11 — INDIVIDUAL AGENT PAGES
    # ══════════════════════════════════════════════════════════════
    AGENTS = [
        {
            "name":     "THOR",
            "subtitle": "The Zone Destroyer  ·  v17a Strategy",
            "color":    THOR_C,
            "role":     "Base Agent",
            "tagline":  "Strikes from above — sells PE when price occupies the key selling zones identified through CPR and pivot analysis.",
            "zone":     "TC to PDH  /  R1 to R2  /  CPR Interior zones",
            "opt":      "PE Sell  (bearish price zone bias)",
            "day_type": "Base Days only",
            "lots":     "Score-based: 1–3 lots  (conviction scoring framework)",
            "insight":  "THOR is the highest-contribution agent — 309 trades and Rs.8.57L P&L over 5 years. The zone classification produces 16 distinct configurations based on where the day's open falls relative to CPR and pivot levels. EMA bias confirms direction.",
            "risk_note":"Hard SL rate 10.4% — managed by conviction score (low-score days get 1 lot only).",
            "rules": [
                "All zones computed from previous day OHLC before market open — zero forward bias.",
                "EMA (20-day) confirms directional bias at market open.",
                "Entry strictly at next candle open + 2 seconds after signal.",
                "Score filter: low-conviction days reduce lot size to minimum.",
                "PE sell filtered when futures basis is 50–100 pts (adverse premium structure).",
                "Maximum 1 THOR trade per day.",
            ],
        },
        {
            "name":     "HULK",
            "subtitle": "The Breakdown Hammer  ·  cam_l3 Strategy",
            "color":    HULK_C,
            "role":     "Base Agent",
            "tagline":  "Smashes from the Camarilla L3 zone — sells PE when price rallies into the PDH-to-R1 resistance channel.",
            "zone":     "PDH to R1  (Camarilla L3 level)",
            "opt":      "PE Sell",
            "day_type": "Base Days only",
            "lots":     "Score-based: 1–3 lots",
            "insight":  "HULK is the highest win-rate base agent at 88.5%, with 84.6% of trades exiting at target. The Camarilla L3 level provides a mathematically precise resistance zone derived from previous-day range and close.",
            "risk_note":"Hard SL rate 7.7% — among the lowest of base agents.",
            "rules": [
                "CAM L3 level computed from previous day High, Low, Close before market open.",
                "Price must rally into the zone, not gap through it.",
                "Confirmation via multi-timeframe alignment before entry.",
                "Score of at least 2 confluence features required for standard lot size.",
                "Entry: next candle + 2 seconds after signal confirmation.",
                "Maximum 1 HULK trade per day.",
            ],
        },
        {
            "name":     "IRON MAN",
            "subtitle": "The Precision Sniper  ·  cam_h3 / iv2_r1 / iv2_r2",
            "color":    IRON_C,
            "role":     "Base Agent",
            "tagline":  "High-precision targeting at upper resistance — sells PE at R1, R2, and CAM H3 levels where supply is concentrated.",
            "zone":     "R1  /  R2  /  CAM H3  (upper resistance band)",
            "opt":      "PE Sell  (at resistance)  ·  CE Sell  (breakdown zones)",
            "day_type": "Base Days only",
            "lots":     "Score-based: 1–3 lots",
            "insight":  "IRON MAN covers three critical upper resistance levels. The cam_h3+tc_to_pdh combination is explicitly excluded — a structural mismatch identified in backtesting where the combination produced adverse risk-reward.",
            "risk_note":"Hard SL rate 3.2% — lowest of all base agents at resistance levels.",
            "rules": [
                "R1 = 2×Pivot − Previous Day Low. R2 = Pivot + Previous Day Range.",
                "CAM H3 = Previous Day Close + Range × 0.275.",
                "cam_h3 + tc_to_pdh zone combination is excluded (structural conflict).",
                "Futures basis must be within acceptable range for PE sells.",
                "Score-based lot sizing applies — no trade below minimum conviction.",
                "Entry: next candle open + 2 seconds.",
            ],
        },
        {
            "name":     "CAPTAIN",
            "subtitle": "The Reliable Soldier  ·  iv2_pdl Strategy",
            "color":    CAP_C,
            "role":     "Base Agent",
            "tagline":  "Consistent and disciplined — level-based entries at PDL zone. One lot, every time. No exceptions.",
            "zone":     "Previous Day Low (PDL)",
            "opt":      "CE Sell  (at PDL breakdown)",
            "day_type": "Base Days only",
            "lots":     "Fixed: 1 lot always",
            "insight":  "CAPTAIN targets the Previous Day Low — a key support-turned-resistance level. With 23 trades and Rs.18,561 P&L, this is the smallest-volume agent. It fires only when price touches the PDL with iv2 pattern confirmation. No conviction scoring — always 1 lot.",
            "risk_note":"Hard SL rate 0.0% — no hard stop hits in 5 years of backtest.",
            "rules": [
                "PDL = Previous Day Low (fixed level, computed pre-market).",
                "iv2 pattern confirmation required at PDL level.",
                "Entry between 09:16 and 15:15 only.",
                "Fixed 1-lot position — no conviction score adjustment.",
                "No re-entry if CAPTAIN has already traded today.",
                "EOD exit at 15:20 if target not reached.",
            ],
        },
        {
            "name":     "SPIDER-MAN",
            "subtitle": "The Web Trap  ·  CRT (Candle Range Theory)",
            "color":    SPIDER_C,
            "role":     "Blank Day Agent",
            "tagline":  "Sets the trap — price sweeps above TC or R1, lures bulls, then snaps back below. CE sell on the reversal.",
            "zone":     "TC  /  R1  (sweep-and-reverse pattern)",
            "opt":      "CE Sell  (bearish reversal after liquidity sweep)",
            "day_type": "Blank Days only  (no base signal that day)",
            "lots":     "Fixed: 1 lot",
            "insight":  "The CRT pattern is a 3-candle trap structure on the 15-minute chart, confirmed by a 5-minute Heiken Ashi setup. It identifies false breakout moves above key resistance where institutions sweep retail stop-losses before reversing.",
            "risk_note":"Hard SL rate 3.7% — IB filter eliminates most adverse entries.",
            "rules": [
                "Only fires on blank days when base agents (THOR/HULK/IRON MAN/CAPTAIN) have not signaled.",
                "15-minute and 5-minute multi-timeframe confirmation required.",
                "IB filter: if Initial Balance (09:15–09:45) already expanded up before entry → SKIP.",
                "Futures basis filter: −50 to +100 pts range required.",
                "Signal must be confirmed before 12:00 PM.",
                "Fixed 1-lot position.",
            ],
        },
        {
            "name":     "BLACK WIDOW",
            "subtitle": "The Silent Reversal  ·  MRC (Mean Reversion Concept)",
            "color":    WIDOW_C,
            "role":     "Blank Day Agent",
            "tagline":  "Strikes from the shadows — PE sell when market bounces off the 38.2% Fibonacci retracement of previous-day range.",
            "zone":     "l_382 = PDH − Range × 0.382  (Fibonacci mean reversion zone)",
            "opt":      "PE Sell  (bullish reversal at Fibonacci support)",
            "day_type": "Blank Days only  (no base signal that day)",
            "lots":     "Fixed: 2 lots  (WR 80.6% justifies double position size)",
            "insight":  "BLACK WIDOW is the highest win-rate agent in the system at 80.6%. The Fibonacci l_382 level is a precise mean-reversion zone where institutions accumulate long positions. The 2-lot sizing was approved after rigorous risk analysis: hard SL rate 4.7%, worst 2-lot loss Rs.11,297, max drawdown unchanged.",
            "risk_note":"Hard SL rate 4.7% (8 hard SLs in 170 trades). MRC CE trades permanently excluded — net negative over 5 years.",
            "rules": [
                "Only fires on blank days when no base agent has signaled.",
                "Previous day range must exceed 50 points (volatility filter).",
                "5-minute Heiken Ashi confirmation required at l_382 zone.",
                "Signal window: 09:15 to 12:00 PM only.",
                "MRC CE (below l_618) trades are permanently excluded.",
                "Fixed 2-lot position — approved on WR ≥ 75% threshold.",
            ],
        },
        {
            "name":     "HAWKEYE",
            "subtitle": "The Precision Re-entry  ·  S4 Second Trade",
            "color":    HAWK_C,
            "role":     "Re-entry Agent",
            "tagline":  "Never misses the second shot — re-enters in the same direction after the base trade hits target, on a defined pullback.",
            "zone":     "Same as triggering base agent  (post-target pullback level)",
            "opt":      "Same as base trade  (CE or PE)",
            "day_type": "Base Days only  (after base agent hits 30% target)",
            "lots":     "Fixed: 1 lot",
            "insight":  "HAWKEYE activates only when a base agent hits the 30% target before 13:30, then waits for the option premium to pull back to 60–75% of original entry. This captures the continuation move in the same direction. 196 trades, 71.4% WR, Rs.1.62L contribution.",
            "risk_note":"Hard SL rate 7.1% — acceptable given the second-trade continuation nature.",
            "rules": [
                "Triggers only when a base agent exits with 'target' reason before 13:30.",
                "Wait for option premium to pull back to 60–75% of original entry price.",
                "Re-entry in same direction as original base trade.",
                "Only 1 HAWKEYE re-entry per day.",
                "Fixed 1-lot — no conviction score adjustment for re-entries.",
                "EOD exit at 15:20 if target not reached.",
            ],
        },
    ]

    for agent in AGENTS:
        name  = agent["name"]
        color = agent["color"]
        clr   = colors.HexColor(color)

        # ── Header bar ──
        hdr = Table([[
            Paragraph(f"<font color='white'><b>{name}</b></font>", AGENT_T),
            Paragraph(agent["subtitle"],  AGENT_S),
            Paragraph(f"<font color='#F0B90B'>{agent['role']}</font>",
                      S("ar", fontSize=9, textColor=GOLD, fontName="Helvetica-Bold",
                        leading=13, alignment=TA_RIGHT)),
        ]], colWidths=[5*cm, 7*cm, 4*cm])
        hdr.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), clr),
            ("TOPPADDING",    (0,0), (-1,-1), 12),
            ("BOTTOMPADDING", (0,0), (-1,-1), 12),
            ("LEFTPADDING",   (0,0), (-1,-1), 14),
            ("RIGHTPADDING",  (0,0), (-1,-1), 14),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(hdr)
        story.append(Spacer(1, 0.25*cm))

        # ── Emblem + Stats side by side ──
        emb_img  = buf_to_rl_image(emblem_bufs[name], 3.2, 3.2)

        # Get real stats
        if name in ag.index:
            a        = ag.loc[name]
            t_cnt    = int(a["trades"])
            wr_val   = a["wr_pct"]
            total_v  = a["total"]
            avg_v    = a["avg"]
            hsl_pct  = a["hard_sl_pct"]
            tgt_pct  = a["target_pct"]
        else:
            t_cnt, wr_val, total_v, avg_v, hsl_pct, tgt_pct = 0, 0, 0, 0, 0, 0

        wr_color = (GREEN if wr_val >= 75 else
                    BLUE  if wr_val >= 70 else RED)

        gauge_img = buf_to_rl_image(gauge_bufs.get(name, io.BytesIO()), 2.0, 2.0) \
                    if name in gauge_bufs else Spacer(1, 2*cm)

        stat_rows = [
            ["TRADES (5yr)",  str(t_cnt)],
            ["WIN RATE",      f"{wr_val:.1f}%"],
            ["5yr PnL",       f"Rs.{total_v:,.0f}"],
            ["AVG PER TRADE", f"Rs.{avg_v:,.0f}"],
            ["TARGET EXITS",  f"{tgt_pct:.1f}%"],
            ["HARD SL RATE",  f"{hsl_pct:.1f}%"],
        ]
        # ── Top row: emblem (left) + stats table (right) ──────────────
        stat_tbl = Table(
            [[Paragraph(f"<b>{k}</b>",
                        S("sk", fontSize=9, fontName="Helvetica-Bold",
                          textColor=SILVER, leading=13)),
              Paragraph(v,
                        S("sv", fontSize=11, fontName="Helvetica-Bold",
                          textColor=DARK_GREY if k != "WIN RATE" else wr_color,
                          leading=15, alignment=TA_RIGHT))]
             for k, v in stat_rows],
            colWidths=[4.2*cm, 3.5*cm])
        stat_tbl.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [LIGHT_GREY, WHITE]),
            ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
            ("TOPPADDING",     (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
            ("LEFTPADDING",    (0,0), (-1,-1), 8),
            ("RIGHTPADDING",   (0,0), (-1,-1), 8),
            ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
        ]))

        # Gauge sits next to stat table
        gauge_stat_row = Table(
            [[stat_tbl, gauge_img]],
            colWidths=[7.7*cm, 2.5*cm])
        gauge_stat_row.setStyle(TableStyle([
            ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
        ]))

        top_row = Table([[emb_img, gauge_stat_row]],
                        colWidths=[3.4*cm, 12.8*cm])
        top_row.setStyle(TableStyle([
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
        ]))
        story.append(top_row)
        story.append(Spacer(1, 0.25*cm))

        # ── Tagline ──────────────────────────────────────────────────
        story.append(Paragraph(f"<i>{agent['tagline']}</i>",
                      S("tl", fontSize=10, textColor=SILVER, fontName="Helvetica-Oblique",
                        leading=15)))
        story.append(Spacer(1, 0.2*cm))

        # ── Zone / Trade / Days / Lots table ─────────────────────────
        details = [
            ("Zone",  agent["zone"]),
            ("Trade", agent["opt"]),
            ("Days",  agent["day_type"]),
            ("Lots",  agent["lots"]),
        ]
        det_tbl = Table(details, colWidths=[1.8*cm, 14.4*cm])
        det_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,-1), MID_GREY),
            ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
            ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ]))
        story.append(det_tbl)
        story.append(Spacer(1, 0.2*cm))

        # ── Insight + Risk ────────────────────────────────────────────
        story.append(Paragraph(
            f"<b>Performance Insight:</b> {agent['insight']}",
            S("ins", fontSize=9.5, textColor=DARK_GREY, fontName="Helvetica",
              leading=14, alignment=TA_JUSTIFY)))
        story.append(Spacer(1, 0.1*cm))
        story.append(Paragraph(
            f"<b>Risk Note:</b> {agent['risk_note']}",
            S("rn", fontSize=9, textColor=RED, fontName="Helvetica", leading=13)))
        story.append(Spacer(1, 0.2*cm))

        # ── Rules ─────────────────────────────────────────────────────
        story.append(Paragraph(f"<b>{name} Rules &amp; Filters</b>", H3))
        for i, rule in enumerate(agent["rules"], 1):
            story.append(Paragraph(f"  <b>{i}.</b>  {rule}", RULE_B))

        # ── Exit distribution mini table ───────────────────────────────
        if name in ag.index:
            a2 = ag.loc[name]
            n  = int(a2["trades"])
            target_n  = int(a2["target_n"])
            hard_sl_n = int(a2["hard_sl"])
            eod_n     = n - target_n - hard_sl_n - int(a2["target_pct"]*0/100)
            # compute from raw df
            agent_df  = df[df["agent"] == name]
            eod_cnt   = int((agent_df["exit_reason"] == "eod").sum())
            lock_cnt  = int((agent_df["exit_reason"] == "lockin_sl").sum())

            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph("Exit Breakdown", H3))
            ex_dist = [
                ["Exit Type", "Count", "% of Trades", "Typical Outcome"],
                ["Target (-30%)",  str(target_n),  f"{target_n/n*100:.1f}%",  "Full profit booked"],
                ["Lock-in SL",     str(lock_cnt),  f"{lock_cnt/n*100:.1f}%",  "Partial profit locked"],
                ["EOD (15:20)",    str(eod_cnt),   f"{eod_cnt/n*100:.1f}%",   "Variable — time-based exit"],
                ["Hard SL (+100%)",str(hard_sl_n), f"{hard_sl_n/n*100:.1f}%", "Full loss on entry premium"],
            ]
            ex_d_tbl = Table(ex_dist, colWidths=[4*cm, 2*cm, 3*cm, 7.2*cm])
            ex_d_ts = tblstyle(colors.HexColor(color))
            ex_d_ts.add("BACKGROUND", (0,1), (-1,1), GREEN_LIGHT)
            ex_d_ts.add("BACKGROUND", (0,4), (-1,4), RED_LIGHT)
            ex_d_tbl.setStyle(ex_d_ts)
            story.append(ex_d_tbl)

        story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # PAGE: ENTRY RULES
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Universal Entry Rules", H1))
    story.append(hr())

    entry_rules = [
        ("No Forward Bias",
         "All technical levels (CPR, Camarilla, Fibonacci, PDH/PDL, Pivot R1/R2) are computed "
         "from <b>previous day data</b> before market opens at 08:55 AM. "
         "Zero current-day or future data is used at any stage."),
        ("Entry Timing",
         "Every entry is: <b>next candle open + 2 seconds</b> after signal confirmation. "
         "This prevents entering on the signal candle itself. "
         "Intraday window: 09:16 to 15:15."),
        ("IB Filter (SPIDER-MAN)",
         "Initial Balance = NIFTY spot range from 09:15 to 09:45. "
         "If IB has already expanded <b>upward</b> before the entry time, "
         "SPIDER-MAN CE sell is skipped — selling into a confirmed uptrend is adverse."),
        ("Futures Basis Filter",
         "Futures basis = Futures LTP − Spot LTP at 09:15. "
         "SPIDER-MAN: basis must be −50 to +100 pts. "
         "Base PE sells (THOR/HULK): skipped when basis is 50–100 pts (strong bull premium)."),
        ("Blank Day Logic",
         "SPIDER-MAN and BLACK WIDOW <b>only trade on blank days</b> — "
         "days when no base agent (THOR/HULK/IRON MAN/CAPTAIN) fires a signal. "
         "On base-agent days, only the base agent and HAWKEYE are eligible."),
        ("Sequential Only",
         "One active position at a time. A new signal is only considered "
         "<b>after the current trade exits</b> completely. "
         "Exception: HAWKEYE can activate after base agent exits at target."),
        ("Pre-Market Calculation",
         "At 08:55 AM each day: fetch 45-day daily OHLC → compute EMA(20), CPR, "
         "Camarilla, Fibonacci, Pivot levels. Score7 conviction features computed. "
         "Day plan is locked before 09:15."),
    ]
    for title, desc in entry_rules:
        story.append(Paragraph(f"<b>{title}</b>", H3))
        story.append(Paragraph(desc, BODY))
        story.append(Spacer(1, 0.08*cm))
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════
    # EXIT RULES  (continues from Entry Rules)
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Universal Exit Rules", H1))
    story.append(hr())
    story.append(Paragraph(
        "All agents share the same exit framework. Once a position is open, "
        "the rules below execute mechanically — no human override at any stage.", BODY))
    story.append(Spacer(1, 0.3*cm))

    exit_data = [
        ["Exit Type",          "Trigger Condition",                              "Typical P&L",  "Frequency"],
        ["TARGET",             "Option price falls 30% from entry (ep × 0.70)", "Rs.+1,500–6,000", "63.5%"],
        ["TRAIL SL — BE",      "Price declines 25% from entry → SL = entry",    "Rs. ~0",          "Part of 9%"],
        ["TRAIL SL — Lock20",  "Price declines 40% from entry → SL = 80% ep",   "Rs.+300–1,500",   "Part of 9%"],
        ["TRAIL SL — Ride",    "Price declines 60%+ → SL trails at 95% of max", "Rs.+large",        "Part of 9%"],
        ["EOD EXIT",           "Time ≥ 15:20:00 — force close regardless",      "Rs. ±variable",   "20.7%"],
        ["HARD SL",            "Option price doubles (ep × 2.0) — immediate exit","Rs.−4K to −17K", "6.8%"],
    ]
    exit_tbl = Table(exit_data, colWidths=[3.5*cm, 6.5*cm, 3.0*cm, 2.5*cm])
    exit_ts = tblstyle(DARK)
    exit_ts.add("BACKGROUND", (0,1), (-1,1), GREEN_LIGHT)
    exit_ts.add("BACKGROUND", (0,5), (-1,5), LIGHT_GREY)
    exit_ts.add("BACKGROUND", (0,6), (-1,6), RED_LIGHT)
    exit_ts.add("FONTNAME",   (0,1), (0,1),  "Helvetica-Bold")
    exit_ts.add("FONTNAME",   (0,6), (0,6),  "Helvetica-Bold")
    exit_ts.add("TEXTCOLOR",  (0,1), (0,1),  GREEN)
    exit_ts.add("TEXTCOLOR",  (0,6), (0,6),  RED)
    exit_tbl.setStyle(exit_ts)
    story.append(exit_tbl)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Trailing Stop-Loss Mechanics", H2))
    trail_data = [
        ["Entry = Rs.100", "Sell 1 CE or PE at Rs.100"],
        ["Target = Rs.70", "Price falls to Rs.70 → exit, profit = Rs.30 per unit"],
        ["Trail 1: decline 25%", "Price falls to Rs.75 → SL moves to Rs.100 (breakeven locked)"],
        ["Trail 2: decline 40%", "Price falls to Rs.60 → SL moves to Rs.80 (20% locked)"],
        ["Trail 3: decline 60%", "Price falls to Rs.40 → SL trails at 95% of max gain"],
        ["Hard SL = Rs.200",     "Price rises to Rs.200 → immediate exit, full loss capped"],
    ]
    tr_tbl = Table(trail_data, colWidths=[4.5*cm, 11.5*cm])
    tr_tbl.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (1,0), (-1,-1), "Helvetica"),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [LIGHT_GREY, WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND",    (0,1), (-1,1), GREEN_LIGHT),
        ("BACKGROUND",    (0,5), (-1,5), RED_LIGHT),
    ]))
    story.append(tr_tbl)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "<b>5-year exit distribution:</b> Target 63.5% · Trail SL 9.0% · EOD 20.7% · Hard SL 6.8%", BODY))
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════
    # CONVICTION SCORING  (continues from Exit Rules)
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Conviction Scoring — Lot Sizing Framework", H1))
    story.append(hr())
    story.append(Paragraph(
        "For base agents (THOR, HULK, IRON MAN, CAPTAIN), position size is determined by "
        "a <b>7-feature conviction score</b> computed from previous-day data. "
        "Higher conviction = more lots = amplified P&L when the trade is correct.", BODY))
    story.append(Spacer(1, 0.3*cm))

    score_data = [
        ["Feature",         "Condition for 1 Point",                    "Why It Matters"],
        ["VIX OK",          "India VIX < 20-day MA of VIX",             "Low fear environment favors premium sellers"],
        ["CPR Trend",       "Prev close on correct side of CPR",        "Prior day momentum confirms direction"],
        ["Consecutive",     "2 consecutive days close aligned",         "Multi-day momentum adds conviction"],
        ["Gap Aligned",     "Today's open gaps in trade direction",     "Gap = strong institutional intent"],
        ["DTE Sweet Spot",  "Days to expiry: 3 to 5",                   "Theta acceleration zone — max time decay"],
        ["CPR Narrow",      "CPR width 0.10%–0.20% of spot",           "Narrow CPR signals trending day"],
        ["CPR Directional", "CPR midpoint trending in trade direction", "Macro CPR slope confirms setup"],
    ]
    sc_tbl = Table(score_data, colWidths=[3.2*cm, 5.5*cm, 7.3*cm])
    sc_tbl.setStyle(tblstyle(DARK))
    story.append(sc_tbl)
    story.append(Spacer(1, 0.4*cm))

    _lot_h2 = Paragraph("Score to Lot Size Mapping", H2)
    lot_data = [
        ["Score",        "Base Lots", "Inside CPR?", "Final Lots", "Meaning"],
        ["0 – 1",        "1",         "Any",         "1",          "Low conviction — minimum exposure"],
        ["2 – 3",        "2",         "No",          "2",          "Moderate conviction — standard size"],
        ["2 – 3",        "2",         "Yes",         "1",          "Inside CPR penalty (−1 lot)"],
        ["4 – 7",        "3",         "No",          "3",          "High conviction — full position"],
        ["4 – 7",        "3",         "Yes",         "2",          "Inside CPR reduces to 2"],
        ["Score = 6",    "—",         "—",           "SKIP",       "Score==6 excluded (historically adverse)"],
        ["BLACK WIDOW",  "2",         "N/A",         "2",          "Fixed 2-lot — WR 80.6% justifies"],
        ["SPIDER-MAN",   "1",         "N/A",         "1",          "Fixed 1-lot"],
        ["HAWKEYE",      "1",         "N/A",         "1",          "Fixed 1-lot"],
    ]
    lt_tbl = Table(lot_data, colWidths=[2.8*cm, 2.0*cm, 2.5*cm, 2.5*cm, 6.2*cm])
    lt_ts = tblstyle(LIGHT_BLUE)
    lt_ts.add("BACKGROUND", (0,6), (-1,6), RED_LIGHT)
    lt_ts.add("TEXTCOLOR",  (3,6), (3,6), RED)
    lt_ts.add("FONTNAME",   (3,6), (3,6), "Helvetica-Bold")
    lt_tbl.setStyle(lt_ts)
    story.append(KeepTogether([_lot_h2, lt_tbl]))
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════
    # PERFORMANCE DETAIL  (continues after Conviction Scoring)
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Detailed Performance — 5-Year Breakdown", H1))
    story.append(hr())

    story.append(Paragraph("Year-wise Performance", H2))
    yr_table_data = [["Year", "Trades", "Win Rate", "Total P&L (Rs.)", "Avg/Trade (Rs.)", "vs Prior Year"]]
    prev_pnl = None
    for yr, row in yr_g.iterrows():
        trend = ""
        if prev_pnl is not None:
            delta = (row["total"] - prev_pnl) / abs(prev_pnl) * 100
            trend = f"+{delta:.0f}%" if delta >= 0 else f"{delta:.0f}%"
        yr_table_data.append([
            f"{yr}" + (" *" if yr == 2026 else ""),
            str(int(row["trades"])),
            f"{row['wr']*100:.1f}%",
            f"{row['total']:,.0f}",
            f"{row['avg']:,.0f}",
            trend,
        ])
        prev_pnl = row["total"]
    # Total row
    yr_table_data.append([
        "TOTAL", "949", "74.5%", "16,96,299", "1,788", "5yr compound"
    ])
    yr_tbl = Table(yr_table_data, colWidths=[1.5*cm, 2*cm, 2.2*cm, 3.8*cm, 3.5*cm, 3.0*cm])
    yr_ts = tblstyle(DARK)
    yr_ts.add("BACKGROUND", (0,7), (-1,7), LIGHT_BLUE)
    yr_ts.add("FONTNAME",   (0,7), (-1,7), "Helvetica-Bold")
    yr_ts.add("TEXTCOLOR",  (0,7), (-1,7), WHITE)
    yr_tbl.setStyle(yr_ts)
    story.append(yr_tbl)
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph("* 2026 partial year — January through April only.", SMALL))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Monthly P&L — All 58 Months", H2))
    story.append(buf_to_rl_image(monthly_buf, 16, 6.0))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "Green = positive month · Red = negative month · Gold dashed = monthly average (Rs.29,247). "
        "Negative months: 3 of 58 (5.2%)", SMALL))
    story.append(Spacer(1, 0.2*cm))

    # Monthly stats
    monthly_stats = [
        ["Metric", "Value"],
        ["Total months",          "58"],
        ["Positive months",       "55 (94.8%)"],
        ["Negative months",       "3 (5.2%)"],
        ["Average monthly P&L",   "Rs. 29,247"],
        ["Best month",            "Rs. 1,08,030  (March 2026)"],
        ["Worst month",           "Rs. −20,326  (July 2024)"],
        ["Months > Rs.50,000",    "18 (31%)"],
    ]
    ms_tbl = Table(monthly_stats, colWidths=[5*cm, 11*cm])
    ms_tbl.setStyle(tblstyle(DARK_GREY))
    story.append(ms_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════
    # RISK MANAGEMENT  (continues from Performance Detail)
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Risk Management", H1))
    story.append(hr())

    story.append(Paragraph("Capital Requirements", H2))
    cap_data = [
        ["Parameter",           "Value",            "Notes"],
        ["Minimum capital",     "Rs. 5,00,000",     "For 1-lot base trades with adequate margin buffer"],
        ["Margin per lot",      "Rs. ~50,000",      "NIFTY ATM option sell margin (approximate)"],
        ["Max lots at once",    "3 lots",            "High-conviction base trades only"],
        ["Max daily risk",      "Rs. ~17,000",      "1 hard SL on a 3-lot base trade"],
        ["Monthly avg P&L",     "Rs. 29,247",       "58-month trailing average"],
        ["Negative months",     "3 / 58",           "5.2% of months — extremely low"],
        ["Worst single month",  "Rs. −20,326",      "July 2024 — 2 hard SL same week"],
        ["Best single month",   "Rs. 1,08,030",     "March 2026"],
        ["Return on capital",   "~33.9% annualized","Based on Rs.5L capital, 5yr avg Rs.1.70L/yr"],
    ]
    cap_tbl = Table(cap_data, colWidths=[4.5*cm, 4*cm, 7.5*cm])
    cap_tbl.setStyle(tblstyle(DARK_GREY))
    story.append(cap_tbl)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Risk Control Mechanisms", H2))
    risk_controls = [
        ("Hard SL at 2×",    "Option can never lose more than 100% of entry premium. Maximum total loss on any single trade is capped."),
        ("1 Trade Per Day",  "A bad day impacts only 1 trade. No compounding of intraday losses. Capital protected by design."),
        ("Score Filter",     "Low-conviction days trade 1 lot — limiting daily risk exposure automatically."),
        ("Basis Filter",     "PE sells skipped when futures basis 50–100 pts. Avoids adverse premium structure."),
        ("IB Filter",        "CRT CE skipped if IB already expanded upward — avoids selling into confirmed uptrend."),
        ("Blank Day Split",  "Base agents and blank agents never overlap. Risk per day is always 1 strategy's single trade."),
        ("EOD Forced Exit",  "No overnight holding. Fresh slate every day. No gap risk."),
    ]
    for title, desc in risk_controls:
        story.append(Paragraph(f"<b>{title}:</b> {desc}", RULE_B))
        story.append(Spacer(1, 0.08*cm))

    story.append(Spacer(1, 0.3*cm))
    dd_data = [
        ["Metric",               "Value"],
        ["Maximum Drawdown",     "Rs. 46,134  (2.93% of peak equity)"],
        ["Worst DD Period",      "Jul 15–24, 2024 — 4 trades, 2 hard SL in same week"],
        ["Recovery Time",        "Under 2 months in all DD episodes"],
        ["DD > 5% of equity",    "0 occurrences in 5 years"],
        ["Consecutive losses",   "Max 5 in a row (isolated to volatile weeks only)"],
        ["DD > Rs.1,00,000",     "0 occurrences — hard cap from lot-size limits"],
    ]
    dd_tbl = Table(dd_data, colWidths=[5*cm, 11*cm])
    dd_tbl.setStyle(tblstyle(DARK))
    story.append(Paragraph("Drawdown Profile", H2))
    story.append(dd_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════
    # PAPER TRADING PLAN  (follows Risk Management on same page flow)
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Paper Trading Plan — Getting Started", H1))
    story.append(hr())
    story.append(Paragraph(
        "Before going live, FIFTO recommends a <b>30-day paper trading phase</b> to validate "
        "signal detection, entry timing, and exit execution against live market data. "
        "Results are automatically logged to CSV — compare with backtest to confirm "
        "live-to-backtest alignment (expected: WR within ±5%, avg P&L within ±20%).", BODY))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Daily Operational Routine", H2))
    routine = [
        ("08:55 AM",    "Connect broker API. Fetch 45-day NIFTY daily OHLC. Compute all levels: CPR, Camarilla, MRC Fibonacci, EMA20, Pivot R1/R2. Lock day plan."),
        ("09:10 AM",    "Print day summary: TC, BC, R1, R2, CAM L3/H3, MRC l_382, l_618, nearest expiry, score7 conviction features, expected agent."),
        ("09:15 AM",    "Market open. Start real-time NIFTY spot WebSocket. IB tracking begins (tracks 09:15–09:45 range)."),
        ("09:16 AM",    "Base agents (THOR/HULK/IRON MAN/CAPTAIN) go active. Monitor for zone entry signals."),
        ("09:46 AM",    "IB confirmed. SPIDER-MAN and BLACK WIDOW scanners activate (blank days only)."),
        ("On signal",   "Fetch ATM option LTP. Log paper entry. Start option tick monitoring with automated trade manager."),
        ("During trade","Trailing SL updates on every tick — fully automated. No manual action required."),
        ("15:20 PM",    "Force EOD exit if still in trade. Log result to CSV."),
        ("15:30 PM",    "Print daily P&L summary. Compare with backtest benchmark."),
    ]
    rout_tbl = Table([[t, d] for t, d in routine], colWidths=[3*cm, 13*cm])
    rout_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), MID_GREY),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    story.append(rout_tbl)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Angel One SmartAPI — Data Requirements", H2))
    api_data = [
        ["When",          "What",                "API Method",             "Used For"],
        ["08:55 AM",      "45-day NIFTY OHLC",   "getCandleData (ONE_DAY)","CPR, EMA20, PDH/PDL"],
        ["09:15 AM",      "Futures LTP",          "ltpData()",              "Futures basis filter"],
        ["09:15–15:20",   "NIFTY spot ticks",     "WebSocket (token 26000)","IB, 5M/15M candles"],
        ["On signal",     "ATM option LTP",       "ltpData(NFO)",           "Entry price capture"],
        ["On signal",     "Option ticks",         "WebSocket (NFO token)",  "SL/target monitoring"],
        ["Pre-market",    "Instrument master",    "ScripMaster JSON",       "Option token lookup"],
    ]
    api_tbl = Table(api_data, colWidths=[2.5*cm, 3.5*cm, 4.0*cm, 6.0*cm])
    api_tbl.setStyle(tblstyle(DARK))
    story.append(api_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ══════════════════════════════════════════════════════════════
    # QUICK REFERENCE  (continues from Paper Trading)
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Quick Reference Card", H1))
    story.append(hr())

    # Agent quick ref with emblems
    qr_data = [["", "Agent", "Zone", "Opt", "Lots", "WR", "5yr P&L", "Day Type"]]
    for name, clr, ltr, acc in EMBLEMS:
        emb_sm = buf_to_rl_image(emblem_bufs[name], 0.75, 0.75)
        if name in ag.index:
            a = ag.loc[name]
            wr_s  = f"{a['wr_pct']:.1f}%"
            pnl_s = f"Rs.{a['total']:,.0f}"
        else:
            wr_s, pnl_s = "—", "—"
        zone_map = {
            "THOR":        "TC–PDH / R1–R2",
            "HULK":        "PDH–R1 (CAM L3)",
            "IRON MAN":    "R1 / R2 / CAM H3",
            "CAPTAIN":     "PDL (IV2)",
            "SPIDER-MAN":  "TC / R1 sweep",
            "BLACK WIDOW": "l_382 Fibonacci",
            "HAWKEYE":     "Post-target re-entry",
        }
        opt_map = {
            "THOR":"PE","HULK":"PE","IRON MAN":"PE/CE",
            "CAPTAIN":"CE","SPIDER-MAN":"CE","BLACK WIDOW":"PE","HAWKEYE":"Same",
        }
        lots_map = {
            "THOR":"1–3","HULK":"1–3","IRON MAN":"1–3",
            "CAPTAIN":"1","SPIDER-MAN":"1","BLACK WIDOW":"2","HAWKEYE":"1",
        }
        day_map = {
            "THOR":"Base","HULK":"Base","IRON MAN":"Base",
            "CAPTAIN":"Base","SPIDER-MAN":"Blank","BLACK WIDOW":"Blank","HAWKEYE":"Base",
        }
        qr_data.append([
            emb_sm,
            Paragraph(f"<b><font color='{clr}'>{name}</font></b>",
                      S("qn", fontSize=8.5, fontName="Helvetica-Bold",
                        textColor=colors.HexColor(clr), leading=12)),
            zone_map[name], opt_map[name], lots_map[name], wr_s, pnl_s, day_map[name],
        ])
    qr_tbl = Table(qr_data, colWidths=[1.0*cm, 3.0*cm, 3.2*cm, 1.5*cm, 1.2*cm, 1.5*cm, 3.0*cm, 2.0*cm])
    qr_ts = tblstyle(DARK)
    qr_ts.add("ALIGN",  (0,0), (-1,-1), "CENTER")
    qr_ts.add("VALIGN", (0,0), (-1,-1), "MIDDLE")
    qr_ts.add("TOPPADDING",    (0,1), (-1,-1), 3)
    qr_ts.add("BOTTOMPADDING", (0,1), (-1,-1), 3)
    qr_tbl.setStyle(qr_ts)
    story.append(qr_tbl)
    story.append(Spacer(1, 0.4*cm))

    _ex_h2 = Paragraph("Exit Cheat Sheet", H2)
    ex_data = [
        ["Trigger",                   "Action",                    "Result"],
        ["ep × 0.70",                 "EXIT — Target hit",         "+30% of entry premium"],
        ["Max decline ≥ 25%",         "Move SL to entry price",    "Breakeven locked"],
        ["Max decline ≥ 40%",         "Move SL to ep × 0.80",      "20% gain locked"],
        ["Max decline ≥ 60%",         "Trail SL at 95% of decline","Maximum profit locked"],
        ["ep × 2.0 (price doubles)",  "EXIT — Hard SL",            "100% loss on entry"],
        ["Time ≥ 15:20:00",           "EXIT — EOD forced exit",    "Whatever P&L is at 15:20"],
    ]
    ex_tbl = Table(ex_data, colWidths=[5.5*cm, 5.0*cm, 5.5*cm])
    ex_ts = TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), DARK),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("BACKGROUND",    (0,1), (-1,1), GREEN_LIGHT),
        ("TEXTCOLOR",     (0,1), (0,1), GREEN),
        ("BACKGROUND",    (0,5), (-1,5), RED_LIGHT),
        ("TEXTCOLOR",     (0,5), (0,5), RED),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#B0BEC5")),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ])
    ex_tbl.setStyle(ex_ts)
    story.append(KeepTogether([_ex_h2, ex_tbl]))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph(
        "FIFTO v1.0  ·  5-Year Backtest: 2021–2026  ·  949 Trades  ·  "
        "NIFTY Weekly Options  ·  Zero forward look-ahead bias verified", SMALL))

    doc.build(story, onFirstPage=_draw_cover_bg)
    sz = os.path.getsize(OUT_PATH)
    print(f"PDF created: {OUT_PATH}  ({sz//1024} KB,  ~{len(story)} elements)")


if __name__ == "__main__":
    build()
