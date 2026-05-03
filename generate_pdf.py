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
CSV_PATH   = os.path.join(os.path.dirname(__file__),
                         "data", "20260503", "127_all_trades.csv")
OUT_PATH   = "FIFTO_Intraday_Selling_System.pdf"
LOGO_PATH  = os.path.join(os.path.dirname(__file__), "photo_2025-02-24_18-52-35.jpg")

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


def make_cover_equity(df):
    """Minimal equity curve for cover — dark bg, gold line, clean."""
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(10.5, 3.0), dpi=150)
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")

        eq = df["cum_pnl"].values
        x  = np.arange(len(eq))

        ax.fill_between(x, eq, alpha=0.13, color="#F0B90B", linewidth=0)
        ax.plot(x, eq, color="#F0B90B", linewidth=2.4, solid_capstyle="round")
        ax.axhline(0, color="#2D333B", linewidth=0.8)

        # Year dividers + labels
        for yr in [2021, 2022, 2023, 2024, 2025, 2026]:
            mask = df["date"].str[:4] == str(yr)
            idxs = np.where(mask.values)[0]
            if len(idxs):
                ax.axvline(idxs[0], color="#21262D", linewidth=1.0, zorder=0)
                ax.text(idxs[0] + len(x)*0.005, eq.max()*0.05,
                        str(yr), color="#484F58", fontsize=8.5,
                        ha="left", fontfamily="DejaVu Sans")

        # End dot + label
        ax.scatter([len(eq)-1], [eq[-1]], color="#F0B90B", s=55, zorder=5)
        ax.text(len(eq)*0.985, eq[-1]*0.86,
                f"₹{eq[-1]/1e5:.2f}L",
                color="#F0B90B", fontsize=9.5, ha="right",
                fontweight="bold", fontfamily="DejaVu Sans")

        ax.set_xlim(-8, len(eq) + 12)
        ax.set_ylim(-eq.max()*0.07, eq.max()*1.20)
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="#0D1117", dpi=150, pad_inches=0.05)
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
    canvas.setFillColor(GOLD)
    canvas.rect(0, H - 6, W, 6, fill=1, stroke=0)
    canvas.rect(0, 0, W, 6, fill=1, stroke=0)
    canvas.restoreState()


def _draw_page_header(canvas, doc):
    """Header + footer for every non-cover page: gold bars, logo, page number."""
    canvas.saveState()
    # Top gold bar
    canvas.setFillColor(GOLD)
    canvas.rect(0, H - 4, W, 4, fill=1, stroke=0)
    # Bottom gold bar
    canvas.rect(0, 0, W, 3, fill=1, stroke=0)
    # Logo top-right corner
    if os.path.exists(LOGO_PATH):
        lw, lh = 2.8*cm, 1.35*cm
        canvas.drawImage(LOGO_PATH,
                         W - lw - 1.8*cm, H - lh - 0.22*cm,
                         width=lw, height=lh,
                         preserveAspectRatio=True, mask="auto")
    # Page number bottom-right
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#90A4AE"))
    canvas.drawRightString(W - 2.0*cm, 0.55*cm, str(doc.page))
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
    story.append(Spacer(1, 0.8*cm))

    story.append(HRFlowable(width="100%", thickness=3, color=GOLD,
                             hAlign="CENTER", spaceAfter=0.5*cm))

    # Logo image (grey bg sits as a "card" on dark background)
    _logo_img = RLImage(LOGO_PATH, width=9.5*cm, height=4.75*cm)
    _logo_img.hAlign = "CENTER"
    story.append(_logo_img)

    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        "Intraday  ·  NIFTY 50 Weekly Options  ·  Fully Mechanical  ·  Zero Overnight Risk",
        S("cv_tag", fontSize=10, textColor=SILVER, alignment=TA_CENTER,
          fontName="Helvetica", leading=15)))

    story.append(Spacer(1, 0.5*cm))

    story.append(HRFlowable(width="85%", thickness=0.8, color=GOLD,
                             hAlign="CENTER", spaceAfter=0.35*cm))

    # ── 6-metric stats (2 rows × 3 cols) ─────────────────────────────────────
    _cL = S("cL", fontSize=8.5, textColor=SILVER, alignment=TA_CENTER,
             fontName="Helvetica-Bold", leading=11)
    _cV = S("cV", fontSize=21,  textColor=GOLD,   alignment=TA_CENTER,
             fontName="Helvetica-Bold", leading=26)
    _sep = colors.HexColor("#2D333B")

    _stats_top = Table(
        [[Paragraph("TOTAL P&amp;L",       _cL),
          Paragraph("WIN RATE",             _cL),
          Paragraph("POSITIVE MONTHS",      _cL)],
         [Paragraph("Rs. 16,96,299",        _cV),
          Paragraph("74.5%",               _cV),
          Paragraph("94.8%",               _cV)]],
        colWidths=[6.0*cm, 4.2*cm, 6.0*cm],
        rowHeights=[0.65*cm, 1.25*cm])
    _stats_top.setStyle(TableStyle([
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("LINEAFTER",     (0,0), (1,-1), 0.6, _sep),
    ]))
    story.append(_stats_top)
    story.append(Spacer(1, 0.25*cm))

    _stats_bot = Table(
        [[Paragraph("MAX DRAWDOWN",         _cL),
          Paragraph("AVG / TRADE",          _cL),
          Paragraph("TRACK RECORD",         _cL)],
         [Paragraph("2.93%",               _cV),
          Paragraph("Rs. 1,788",           _cV),
          Paragraph("5 Years",             _cV)]],
        colWidths=[6.0*cm, 4.2*cm, 6.0*cm],
        rowHeights=[0.65*cm, 1.25*cm])
    _stats_bot.setStyle(TableStyle([
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("LINEAFTER",     (0,0), (1,-1), 0.6, _sep),
    ]))
    story.append(_stats_bot)

    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="85%", thickness=0.8, color=GOLD,
                             hAlign="CENTER", spaceAfter=0.3*cm))

    story.append(Paragraph(
        "Zero forward look-ahead bias  ·  Tick-level signal verification  ·  Fully mechanical execution",
        S("cv_tag2", fontSize=9, textColor=SILVER, alignment=TA_CENTER,
          fontName="Helvetica", leading=14)))
    story.append(Spacer(1, 0.18*cm))
    story.append(Paragraph(
        "NIFTY 50  ·  January 2021 – April 2026  ·  949 Trades  ·  58 Months",
        S("cv_tag3", fontSize=9, textColor=SILVER, alignment=TA_CENTER,
          fontName="Helvetica-Oblique", leading=14)))

    story.append(Spacer(1, 7.5*cm))

    story.append(HRFlowable(width="100%", thickness=2.5, color=GOLD,
                             hAlign="CENTER", spaceAfter=0.4*cm))

    story.append(Paragraph("Confidential  ·  For Authorized Recipients Only",
        S("cv_conf", fontSize=9, textColor=SILVER, alignment=TA_CENTER,
          fontName="Helvetica", leading=13)))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # PAGE 2 — WHY CHOOSE FIFTO
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("Why Choose FIFTO?", H1))
    story.append(hr())
    story.append(Paragraph(
        "FIFTO is a <b>fully mechanical, rules-based intraday trading system</b> for NIFTY 50 "
        "weekly options — designed to generate consistent returns with minimal daily involvement. "
        "Every trade is objective, pre-planned, and executed without discretion. "
        "No charts to read. No decisions to make. The system does the work.", BODY))
    story.append(Spacer(1, 0.25*cm))

    # Capital requirement highlight box
    cap_box = Table(
        [
            [Paragraph("CAPITAL REQUIRED", S("cl", fontSize=9, textColor=GOLD,
                         fontName="Helvetica-Bold", alignment=TA_CENTER, leading=12)),
             Paragraph("EXPECTED MONTHLY RETURN", S("cl", fontSize=9, textColor=GOLD,
                         fontName="Helvetica-Bold", alignment=TA_CENTER, leading=12))],
            [Paragraph("Rs. 5,00,000", S("cv", fontSize=22, textColor=GOLD,
                         fontName="Helvetica-Bold", alignment=TA_CENTER, leading=26)),
             Paragraph("Rs. 29,247 avg", S("cv2", fontSize=22, textColor=GOLD,
                         fontName="Helvetica-Bold", alignment=TA_CENTER, leading=26))],
        ],
        colWidths=[8.1*cm, 8.1*cm],
        rowHeights=[0.7*cm, 1.4*cm])
    cap_box.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), DARK),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("BOX",           (0,0), (-1,-1), 1.2, GOLD),
        ("LINEAFTER",     (0,0), (0,-1), 1.2, GOLD),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(cap_box)
    story.append(Spacer(1, 0.35*cm))

    story.append(Paragraph("What Makes FIFTO Unique", H2))
    differentiators = [
        ("Zero Discretion",
         "Every entry, exit, and position size is determined by a fully mechanical ruleset. "
         "No market opinion. No emotional bias. The same decision is made every day, every time."),
        ("7 Independent Agents, 1 Portfolio",
         "Seven independent trading agents cover different market conditions — bull days, "
         "bear days, ranging days, and post-target continuation. If one agent is not active, "
         "another takes over. The system is always working."),
        ("Proven 5-Year Track Record",
         "Live-comparable backtest across 1,155 trading days from January 2021 to April 2026. "
         "74.5% win rate. Only 3 negative months out of 58. Max drawdown 2.93%. "
         "Not curve-fitted — tested on out-of-sample periods."),
        ("Low-Risk Structure",
         "Option selling with a defined hard stop-loss and automatic trailing stop. "
         "Maximum loss per trade is capped. Profit is locked progressively as the trade moves in your favour. "
         "You never give back a winning trade."),
        ("Minimal Time Commitment",
         "One pre-market calculation at 08:55 AM. One entry. Automated trail. EOD exit at 15:20. "
         "Total active time: under 15 minutes per day."),
        ("No Overnight Risk",
         "Every position exits by 15:20 daily. Zero overnight exposure. "
         "Capital is free every evening. No gap risk, no event risk, no overnight surprises."),
    ]
    for i, (title, desc) in enumerate(differentiators, 1):
        story.append(Paragraph(f"<b>{i}. {title}:</b> {desc}", RULE_B))
        story.append(Spacer(1, 0.1*cm))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("What You Get", H2))
    what_you_get = [
        ["Daily signals — pre-market, ready before 09:15 AM",
         "Fully defined entry, target, and stop-loss for every trade"],
        ["Live performance dashboard — updated daily",
         "Monthly P&L reports with detailed trade log"],
        ["Dedicated support — queries answered same day",
         "System updates as market evolves — no stale rules"],
    ]
    for row in what_you_get:
        wg_row = Table(
            [[Paragraph(f"✔  {row[0]}", S("wg", fontSize=9.5, textColor=DARK_GREY,
                         fontName="Helvetica", leading=14)),
              Paragraph(f"✔  {row[1]}", S("wg2", fontSize=9.5, textColor=DARK_GREY,
                         fontName="Helvetica", leading=14))]],
            colWidths=[8.1*cm, 8.1*cm])
        wg_row.setStyle(TableStyle([
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]))
        story.append(wg_row)

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
        ("THOR",        THOR_C,   "Open-based zone  (multiple configs)", "PE",   "1–3", "74.1%", "Rs.8,56,859", "Base"),
        ("HULK",        HULK_C,   "Resistance zone above prior range",   "PE",   "1–3", "88.5%", "Rs.1,33,188", "Base"),
        ("IRON MAN",    IRON_C,   "Upper resistance band  (3 levels)",   "PE/CE","1–3", "73.0%", "Rs.1,43,107", "Base"),
        ("CAPTAIN",     CAP_C,    "Prior session support level",         "CE/PE","1",   "60.9%", "Rs.18,561",   "Base"),
        ("SPIDER-MAN",  SPIDER_C, "False breakout reversal zone",        "CE",   "1",   "69.9%", "Rs.74,935",   "Blank"),
        ("BLACK WIDOW", WIDOW_C,  "Statistical mean-reversion zone",     "PE",   "2",   "80.6%", "Rs.3,07,346", "Blank"),
        ("HAWKEYE",     HAWK_C,   "Post-target continuation",            "Same", "1",   "71.4%", "Rs.1,62,302", "Re-entry"),
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
                   colWidths=[1.2*cm, 3.0*cm, 3.6*cm, 1.5*cm, 1.2*cm, 1.5*cm, 3.0*cm, 2.0*cm])
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

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # PAGES 5–11 — INDIVIDUAL AGENT PAGES
    # ══════════════════════════════════════════════════════════════
    AGENTS = [
        {
            "name":     "THOR",
            "subtitle": "The Zone Destroyer  ·  Base Agent",
            "color":    THOR_C,
            "role":     "Base Agent",
            "tagline":  "Strikes at the open — reads where the market has positioned itself at the start of the session and sells premium in the dominant direction.",
            "zone":     "Multiple intraday zones — identified at market open",
            "opt":      "PE Sell",
            "day_type": "Base Days only",
            "lots":     "Conviction-based: 1–3 lots",
            "insight":  "THOR is the highest-contribution agent — 309 trades and Rs.8.57L P&L over 5 years. Fires once at a fixed time each morning, positioning decisively in the direction the session has already committed to. No intraday re-assessment needed.",
            "risk_note":"Hard SL rate 10.4% — managed by conviction scoring which restricts lot size on lower-confidence days.",
            "why_bullets": [
                "Highest P&L contribution in the system — Rs.8,56,859 over 5 years.",
                "74.1% win rate across 309 trades over 5 years.",
                "More conviction = larger position — size is earned, not guessed.",
                "Fires once per day at a fixed pre-planned time. No ambiguity.",
                "Consistent performer across all market regimes — bull, bear, and sideways.",
            ],
        },
        {
            "name":     "HULK",
            "subtitle": "The Breakdown Hammer  ·  Base Agent",
            "color":    HULK_C,
            "role":     "Base Agent",
            "tagline":  "Smashes into resistance — sells premium at a key overhead resistance zone where the market historically stalls and reverses.",
            "zone":     "Key resistance zone above prior session range",
            "opt":      "PE Sell",
            "day_type": "Base Days only",
            "lots":     "Conviction-based: 1–3 lots",
            "insight":  "HULK is the highest win-rate base agent at 88.5%, with 84.6% of all trades exiting at the full profit target. The entry zone is derived entirely from prior-session data — the level is known before the market opens.",
            "risk_note":"Hard SL rate 7.7% — among the lowest of all base agents.",
            "why_bullets": [
                "88.5% win rate — highest of all base agents.",
                "84.6% of trades exit at full target — full premium captured.",
                "Entry zone known before market opens. No intraday guesswork.",
                "Hard SL rate only 7.7% — losses are rare and fully defined.",
                "Reliable across all 5 years — no cherry-picked periods.",
            ],
        },
        {
            "name":     "IRON MAN",
            "subtitle": "The Precision Sniper  ·  Base Agent",
            "color":    IRON_C,
            "role":     "Base Agent",
            "tagline":  "High-precision targeting — sells premium at upper resistance levels where institutional supply consistently overwhelms retail demand.",
            "zone":     "Upper resistance band — three precision levels",
            "opt":      "PE Sell  ·  CE Sell  (by setup)",
            "day_type": "Base Days only",
            "lots":     "Conviction-based: 1–3 lots",
            "insight":  "IRON MAN covers three critical upper resistance levels with the lowest hard SL rate of all base agents at 3.2%. Precision entry zones are pre-calculated daily — the system knows exactly where to act before the market opens.",
            "risk_note":"Hard SL rate 3.2% — lowest of all base agents. Resistance entries protect capital.",
            "why_bullets": [
                "3.2% hard SL rate — the safest base agent by loss frequency.",
                "73.0% win rate covering three distinct resistance levels.",
                "Handles both PE and CE sells — adapts to the day's structure.",
                "Zero discretion — entry zone and lot size are fully pre-determined.",
                "Rs.1.43L contribution over 5 years with tightly controlled drawdown.",
            ],
        },
        {
            "name":     "CAPTAIN",
            "subtitle": "The Reliable Soldier  ·  Base Agent",
            "color":    CAP_C,
            "role":     "Base Agent",
            "tagline":  "Consistent and disciplined — targets a key prior-session support level that has converted to resistance. One lot, every time.",
            "zone":     "Prior session support — converted to resistance",
            "opt":      "CE Sell  ·  PE Sell  (by direction)",
            "day_type": "Base Days only",
            "lots":     "Fixed: 1 lot",
            "insight":  "CAPTAIN targets the previous session's floor — a level that routinely becomes resistance when revisited. With 23 trades and Rs.18,561 P&L, this is a precision agent that fires infrequently but cleanly. Zero hard SL hits in 5 years.",
            "risk_note":"Hard SL rate 0.0% — no hard stop hits in the entire 5-year backtest period.",
            "why_bullets": [
                "0.0% hard SL rate — not a single hard stop hit in 5 years.",
                "60.9% win rate at a level that rarely gives false signals.",
                "Simple, fixed 1-lot sizing — no complexity, no surprises.",
                "Fires rarely, but with high conviction every time.",
                "A clean complement to the higher-frequency agents.",
            ],
        },
        {
            "name":     "SPIDER-MAN",
            "subtitle": "The Web Trap  ·  Blank Day Agent",
            "color":    SPIDER_C,
            "role":     "Blank Day Agent",
            "tagline":  "Sets the trap — identifies false breakout moves where retail momentum chases price into resistance, then sells the reversal.",
            "zone":     "False breakout reversal zone — identified intraday",
            "opt":      "CE Sell  (post-reversal)",
            "day_type": "Blank Days only  (base agents not active)",
            "lots":     "Fixed: 1 lot",
            "insight":  "SPIDER-MAN activates on blank days — days when no base agent fires. It identifies a specific intraday trap structure where retail participants are caught on the wrong side, then sells premium into the reversal. 69.9% WR, Rs.74,935 over 5 years.",
            "risk_note":"Hard SL rate 3.7% — robust filters eliminate most adverse entry conditions.",
            "why_bullets": [
                "Activates on blank days — the system is never idle.",
                "69.9% win rate on a setup that triggers only when conditions are optimal.",
                "Hard SL rate 3.7% — multiple pre-entry filters protect capital.",
                "Rs.74,935 P&L on blank days alone — pure additional return.",
                "Strict entry criteria — no signal unless setup quality is high.",
            ],
        },
        {
            "name":     "BLACK WIDOW",
            "subtitle": "The Silent Reversal  ·  Blank Day Agent",
            "color":    WIDOW_C,
            "role":     "Blank Day Agent",
            "tagline":  "Strikes from the shadows — positions PE sell at a statistical mean-reversion zone where price consistently bounces with high precision.",
            "zone":     "Statistical mean-reversion zone — derived from prior range",
            "opt":      "PE Sell",
            "day_type": "Blank Days only  (base agents not active)",
            "lots":     "Fixed: 2 lots  (justified by 80.6% win rate)",
            "insight":  "BLACK WIDOW is the highest win-rate agent in the entire system at 80.6%. The entry zone is a statistically robust mean-reversion level derived from prior-session data. The 2-lot position size was validated after full risk analysis — hard SL rate 4.7%, worst loss Rs.11,297, max drawdown unchanged.",
            "risk_note":"Hard SL rate 4.7% over 170 trades. One-direction trades only — the losing side is permanently excluded.",
            "why_bullets": [
                "80.6% win rate — highest of all 7 agents in the FIFTO system.",
                "Rs.3,07,346 P&L from blank days alone — a complete strategy in itself.",
                "2-lot sizing validated — high win rate earns the larger position.",
                "Losing trade direction permanently excluded — only the profitable side traded.",
                "Hard SL rate 4.7% with worst single loss Rs.11,297 — fully manageable.",
            ],
        },
        {
            "name":     "HAWKEYE",
            "subtitle": "The Precision Re-entry  ·  Re-entry Agent",
            "color":    HAWK_C,
            "role":     "Re-entry Agent",
            "tagline":  "Never misses the second shot — re-enters in the same direction after the base trade hits full target, capturing the continuation move.",
            "zone":     "Continuation zone — same direction as base trade",
            "opt":      "Same as triggering base trade",
            "day_type": "Base Days only  (after base agent hits target)",
            "lots":     "Fixed: 1 lot",
            "insight":  "HAWKEYE activates only when a base agent hits the 30% target before 13:30. It waits for a defined pullback, then re-enters for the continuation move in the same direction. 196 trades, 71.4% WR, Rs.1.62L additional contribution.",
            "risk_note":"Hard SL rate 7.1% — acceptable given the secondary re-entry nature of the trade.",
            "why_bullets": [
                "Extracts a second profit from winning days — pure incremental return.",
                "71.4% win rate with 196 trades — statistically robust re-entry.",
                "Rs.1,62,302 P&L that would otherwise be left on the table.",
                "Only activates on high-quality days when a full target was already hit.",
                "One re-entry per day maximum — disciplined, not greedy.",
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

        # ── Why This Agent ────────────────────────────────────────────
        story.append(Paragraph(f"<b>Why {name}?</b>", H3))
        for bullet in agent["why_bullets"]:
            story.append(Paragraph(f"  ✔  {bullet}", RULE_B))

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
    story.append(Paragraph("How FIFTO Protects Your Capital", H1))
    story.append(hr())

    entry_rules = [
        ("Pre-Market Plan — No Intraday Decisions",
         "All levels and trade parameters are calculated <b>before the market opens at 09:15 AM</b>. "
         "By the time trading begins, the system knows exactly what to look for, where to enter, "
         "and what the target and stop-loss are. Zero real-time decision-making required."),
        ("Precision Entry — No Chasing",
         "Every entry fires at a precisely defined moment after signal confirmation. "
         "The system never chases price. If the entry condition is not met exactly, no trade is taken. "
         "Intraday trading window: 09:16 to 15:15."),
        ("Multi-Layer Signal Filters",
         "Each agent applies multiple independent filters before a trade is allowed. "
         "Market structure, momentum conditions, and risk parameters must all align simultaneously. "
         "A partial setup = no trade. Discipline is built into the system architecture."),
        ("Conviction-Based Position Sizing",
         "Position size (1–3 lots) is determined automatically by how many confirmation factors align. "
         "High-conviction days get full size. Low-conviction days get minimum size. "
         "The system never over-commits on weak setups."),
        ("One Trade at a Time",
         "Only one active position exists at any moment. No averaging down, no simultaneous bets. "
         "HAWKEYE is the only exception — it can activate <i>after</i> the base agent has already exited at target. "
         "Capital is never split across multiple live positions."),
        ("No Overnight Risk — Ever",
         "Every position exits at 15:20 daily. Absolutely no overnight exposure. "
         "Whatever happens after market close — earnings, geopolitical events, global cues — "
         "cannot affect open FIFTO positions, because there are none."),
        ("Daily Performance Tracking",
         "Every trade is logged with entry price, exit price, reason, P&L, and agent. "
         "Monthly summaries are generated automatically. Full transparency — "
         "every winning trade and every losing trade is documented."),
    ]
    for title, desc in entry_rules:
        story.append(Paragraph(f"<b>{title}</b>", H3))
        story.append(Paragraph(desc, BODY))
        story.append(Spacer(1, 0.08*cm))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # EXIT RULES
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
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # CONVICTION SCORING
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
    story.append(Spacer(1, 0.35*cm))

    # ── Live execution boosts (on top of conviction sizing) ───────────────────
    story.append(Paragraph("Live Execution Boosts (applied on top of conviction size)", H2))
    story.append(Paragraph(
        "Two additional +1 lot boosts are applied at execution time before every entry, "
        "on top of the conviction score. They are <b>not</b> part of the base backtest — "
        "they represent validated improvements from 148_combined_backtest (58 months).", BODY))
    story.append(Spacer(1, 0.2*cm))
    boost_data = [
        ["Boost",            "Condition",                                    "Delta/Month", "Logic"],
        ["DTE ≤ 1",          "Trade day is expiry or day-before expiry",     "+Rs. 3,886",  "Theta decays fastest in last 24h — same premium, higher probability"],
        ["Basis S3",         "|Futures basis| > 50 pts AND direction aligned","+ Rs. 1,311", "Strong basis = carry conviction — PE sell when basis > 50, CE sell when < −50"],
    ]
    boost_tbl = Table(boost_data, colWidths=[2.2*cm, 5.5*cm, 2.8*cm, 6.5*cm])
    b_ts = tblstyle(DARK_GREY)
    b_ts.add("BACKGROUND", (0,1), (-1,2), GREEN_LIGHT)
    boost_tbl.setStyle(b_ts)
    story.append(boost_tbl)
    story.append(Paragraph(
        "Both boosts cap at 3 lots total. Combined (joint backtest): +Rs.4,734/mo (+16.2% over base).", SMALL))
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
        ("08:55 AM",    "Connect broker API. Fetch NIFTY daily OHLC. Compute all pre-market levels and conviction score. Lock day plan before market opens."),
        ("09:10 AM",    "Print day summary: all levels, nearest expiry, conviction score, expected agent for the day."),
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
    # SYSTEM v1.1 — LIVE IMPROVEMENTS
    # ══════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("System v1.1 — Validated Live Improvements", H1))
    story.append(hr())
    story.append(Paragraph(
        "Three bias-free improvements verified against the full 58-month combined backtest "
        "(January 2021 – April 2026, 148_combined_backtest.py). All three use only "
        "<b>pre-market or real-time data</b> available at trade entry — zero look-ahead. "
        "Combined uplift verified in joint backtest: <b>+Rs.6,374/month "
        "(+21.8% over base Rs.29,247/month)</b>. "
        "Note: individual improvement figures below are standalone estimates; the joint "
        "backtest gives the true combined figure (Rs.35,621/mo) after accounting for "
        "3-lot cap interactions between DTE and Basis boosts.", BODY))
    story.append(Spacer(1, 0.3*cm))

    # Impact summary table
    impact_data = [
        ["Improvement",           "Type",         "Standalone/Month", "Notes"],
        ["DTE ≤ 1 Lot Boost",     "Lot sizing",   "+Rs. 3,886",       "Add 1 lot on expiry / day-before"],
        ["Basis S3 Lot Boost",    "Lot sizing",   "+Rs. 1,311",       "|Futures basis| > 50 pts, direction aligned"],
        ["Contra Trade",          "New signal",   "+Rs. 1,640",       "Sell opposite option after hard_sl pullback"],
        ["COMBINED (joint backtest)", "—",        "+Rs. 6,374",       "+21.8% vs base — verified in 148_combined_backtest"],
    ]
    imp_tbl = Table(impact_data, colWidths=[4.5*cm, 2.5*cm, 2.8*cm, 6.2*cm])
    imp_ts = tblstyle(DARK)
    imp_ts.add("BACKGROUND", (0,4), (-1,4), LIGHT_BLUE)
    imp_ts.add("FONTNAME",   (0,4), (-1,4), "Helvetica-Bold")
    imp_ts.add("TEXTCOLOR",  (0,4), (-1,4), WHITE)
    imp_ts.add("BACKGROUND", (0,1), (-1,3), GREEN_LIGHT)
    imp_tbl.setStyle(imp_ts)
    story.append(imp_tbl)
    story.append(Spacer(1, 0.45*cm))

    # ── Improvement 1: DTE Boost ──────────────────────────────────────────────
    story.append(Paragraph("1.  DTE ≤ 1 Lot Boost", H2))
    story.append(Paragraph(
        "On <b>expiry day (DTE = 0)</b> and <b>the day before expiry (DTE = 1)</b>, "
        "one additional lot is added to the trade (capped at 3 lots total). "
        "Theta decay accelerates sharply in the final 24 hours — the same option premium "
        "collected decays faster, making each unit of risk proportionally more rewarding.", BODY))
    story.append(Spacer(1, 0.1*cm))
    dte_data = [
        ["Metric",           "Value"],
        ["Condition",        "DTE ≤ 1 (computed from nearest weekly expiry)"],
        ["Action",           "+1 lot added (max 3 lots)"],
        ["Trades affected",  "~15–20% of all trades fall on expiry or day-before"],
        ["Backtest delta",   "+Rs. 3,886 / month  (+Rs. 2,25,388 over 58 months)"],
        ["Risk change",      "None — same hard SL %, larger absolute amount at risk"],
    ]
    dte_tbl = Table(dte_data, colWidths=[3.5*cm, 12.5*cm])
    dte_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), MID_GREY),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [LIGHT_GREY, WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("BACKGROUND",    (0,4), (-1,4), GREEN_LIGHT),
    ]))
    story.append(dte_tbl)
    story.append(Spacer(1, 0.35*cm))

    # ── Improvement 2: Basis S3 Boost ────────────────────────────────────────
    story.append(Paragraph("2.  Futures Basis S3 Lot Boost", H2))
    story.append(Paragraph(
        "When the <b>NIFTY futures basis exceeds 50 points in magnitude AND is aligned "
        "with the trade direction</b>, one additional lot is added. A large positive basis "
        "(spot below futures) favours PE sellers — institutions are bidding futures up, "
        "reducing downside risk. A large negative basis favours CE sellers.", BODY))
    story.append(Spacer(1, 0.1*cm))
    basis_data = [
        ["Metric",           "Value"],
        ["Condition",        "|Futures basis| > 50 pts  AND  direction aligned"],
        ["Direction rule",   "PE sell: basis > +50  |  CE sell: basis < −50"],
        ["Action",           "+1 lot added (max 3 lots)"],
        ["Backtest delta",   "+Rs. 1,311 / month  (+Rs. 76,038 over 58 months)"],
        ["Basis fetch time", "09:15 AM — computed once per day from live futures LTP"],
    ]
    basis_tbl = Table(basis_data, colWidths=[3.5*cm, 12.5*cm])
    basis_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), MID_GREY),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [LIGHT_GREY, WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("BACKGROUND",    (0,4), (-1,4), GREEN_LIGHT),
    ]))
    story.append(basis_tbl)
    story.append(Spacer(1, 0.35*cm))

    # ── Improvement 3: Contra Trade ───────────────────────────────────────────
    story.append(Paragraph("3.  Contra Trade — Post Hard-SL Recovery Entry", H2))
    story.append(Paragraph(
        "When a trade exits via <b>hard_sl</b> (option doubles), the system immediately "
        "begins monitoring the NIFTY spot price. When spot <b>pulls back within 30 points "
        "of the spot price at the time of the hard_sl exit</b>, a new trade is entered "
        "in the <b>opposite direction</b> — the market has reversed, confirming the "
        "initial move was a false spike. Entry cutoff: 14:00.", BODY))
    story.append(Spacer(1, 0.1*cm))
    contra_data = [
        ["Metric",           "Value"],
        ["Trigger",          "Hard_sl exit — option premium doubled from entry"],
        ["Entry condition",  "Spot pulls back to within 30 pts of hard_sl exit spot (≤ 14:00)"],
        ["Direction",        "PE hard_sl → sell CE  |  CE hard_sl → sell PE"],
        ["Strike",           "ATM at time of contra entry"],
        ["Lots",             "1 lot (+ DTE / Basis boosts if applicable)"],
        ["Exit rules",       "Identical to base trade: 30% target, 2× hard SL, trail SL, EOD"],
        ["Historical stats", "59 / 65 triggered (90.8%)  |  WR 91.5%  |  +Rs.1,640/month"],
        ["Backtest delta",   "+Rs. 95,111 total over 58 months"],
    ]
    contra_tbl = Table(contra_data, colWidths=[3.5*cm, 12.5*cm])
    contra_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), MID_GREY),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [LIGHT_GREY, WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("BACKGROUND",    (0,7), (-1,8), GREEN_LIGHT),
    ]))
    story.append(contra_tbl)
    story.append(Spacer(1, 0.35*cm))

    story.append(Paragraph(
        "<b>Why contra works:</b> A hard_sl means the market moved sharply against the trade. "
        "In most cases (91.5% historically) the sharp move is temporary — the spot reverts "
        "to the zone it spiked from. Selling the opposite option after the pullback captures "
        "this mean-reversion with the same mechanical exit rules, requiring zero human judgment.",
        S("cbody", fontSize=9.5, textColor=DARK_GREY, fontName="Helvetica",
          leading=14, alignment=TA_JUSTIFY, leftIndent=8, rightIndent=8,
          borderPad=6)))
    story.append(Spacer(1, 0.4*cm))

    # Combined v1.1 metrics box
    v11_box = Table(
        [[Paragraph("BASE SYSTEM (v1.0)", S("v10l", fontSize=9, textColor=GOLD,
                     fontName="Helvetica-Bold", alignment=TA_CENTER, leading=12)),
          Paragraph("v1.1 IMPROVEMENTS", S("v11l", fontSize=9, textColor=GOLD,
                     fontName="Helvetica-Bold", alignment=TA_CENTER, leading=12)),
          Paragraph("COMBINED (v1.1)", S("v11cl", fontSize=9, textColor=GOLD,
                     fontName="Helvetica-Bold", alignment=TA_CENTER, leading=12))],
         [Paragraph("Rs. 29,247 / month", S("v10v", fontSize=16, textColor=WHITE,
                     fontName="Helvetica-Bold", alignment=TA_CENTER, leading=20)),
          Paragraph("+Rs. 6,374 / month", S("v11v", fontSize=16, textColor=GOLD,
                     fontName="Helvetica-Bold", alignment=TA_CENTER, leading=20)),
          Paragraph("Rs. 35,621 / month", S("v11cv", fontSize=16, textColor=GOLD,
                     fontName="Helvetica-Bold", alignment=TA_CENTER, leading=20))],
        ],
        colWidths=[5.4*cm, 5.4*cm, 5.4*cm],
        rowHeights=[0.7*cm, 1.3*cm])
    v11_box.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), DARK),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("BOX",           (0,0), (-1,-1), 1.2, GOLD),
        ("LINEAFTER",     (0,0), (1,-1),  0.8, colors.HexColor("#2D333B")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(v11_box)

    # ══════════════════════════════════════════════════════════════
    # QUICK REFERENCE
    # ══════════════════════════════════════════════════════════════
    story.append(PageBreak())
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
            "THOR":        "Open-based zone",
            "HULK":        "Resistance zone",
            "IRON MAN":    "Upper resistance band",
            "CAPTAIN":     "Prior session support",
            "SPIDER-MAN":  "False breakout reversal",
            "BLACK WIDOW": "Mean-reversion zone",
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
        "FIFTO v1.1  ·  5-Year Backtest: 2021–2026  ·  949 Trades  ·  "
        "NIFTY Weekly Options  ·  Zero forward look-ahead bias verified  ·  "
        "v1.1 improvements: DTE boost · Basis boost · Contra trade", SMALL))

    doc.build(story, onFirstPage=_draw_cover_bg, onLaterPages=_draw_page_header)
    sz = os.path.getsize(OUT_PATH)
    print(f"PDF created: {OUT_PATH}  ({sz//1024} KB,  ~{len(story)} elements)")


if __name__ == "__main__":
    build()
