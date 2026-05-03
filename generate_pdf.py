"""
generate_pdf.py — FIFTO Intraday Option Selling System — Client PDF
====================================================================
Generates: FIFTO_Intraday_Selling_System.pdf
"""
import os
from reportlab.lib          import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles   import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units    import cm, mm
from reportlab.platypus     import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums    import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfbase      import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Color Palette ──────────────────────────────────────────────────────────────
DARK        = colors.HexColor("#0D1117")
GOLD        = colors.HexColor("#F0B90B")
BLUE        = colors.HexColor("#1565C0")
LIGHT_BLUE  = colors.HexColor("#1E3A5F")
RED         = colors.HexColor("#C62828")
GREEN       = colors.HexColor("#2E7D32")
SILVER      = colors.HexColor("#607D8B")
WHITE       = colors.white
LIGHT_GREY  = colors.HexColor("#F5F5F5")
MID_GREY    = colors.HexColor("#ECEFF1")
DARK_GREY   = colors.HexColor("#263238")
ACCENT      = colors.HexColor("#FF6F00")

# Marvel agent colors
THOR_COLOR   = colors.HexColor("#1565C0")   # Thor blue
HULK_COLOR   = colors.HexColor("#2E7D32")   # Hulk green
IRON_COLOR   = colors.HexColor("#C62828")   # Iron Man red-gold
CAP_COLOR    = colors.HexColor("#0D47A1")   # Cap dark blue
SPIDER_COLOR = colors.HexColor("#B71C1C")   # Spider-Man red
WIDOW_COLOR  = colors.HexColor("#37474F")   # Black Widow dark
HAWK_COLOR   = colors.HexColor("#6A1B9A")   # Hawkeye purple

OUT_PATH = "FIFTO_Intraday_Selling_System.pdf"

# ── Style sheet ───────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

COVER_TITLE  = S("ct", fontSize=42, textColor=GOLD, alignment=TA_CENTER,
                 fontName="Helvetica-Bold", leading=50, spaceAfter=8)
COVER_SUB    = S("cs", fontSize=18, textColor=WHITE, alignment=TA_CENTER,
                 fontName="Helvetica", leading=26, spaceAfter=6)
COVER_TAG    = S("ctag", fontSize=12, textColor=SILVER, alignment=TA_CENTER,
                 fontName="Helvetica-Oblique", leading=18)

H1 = S("h1", fontSize=22, textColor=GOLD, fontName="Helvetica-Bold",
        leading=28, spaceBefore=14, spaceAfter=6)
H2 = S("h2", fontSize=15, textColor=LIGHT_BLUE, fontName="Helvetica-Bold",
        leading=20, spaceBefore=10, spaceAfter=4)
H3 = S("h3", fontSize=12, textColor=DARK_GREY, fontName="Helvetica-Bold",
        leading=16, spaceBefore=6, spaceAfter=3)
BODY = S("body", fontSize=10, textColor=DARK_GREY, fontName="Helvetica",
          leading=15, spaceBefore=3, spaceAfter=3, alignment=TA_JUSTIFY)
SMALL = S("small", fontSize=9, textColor=SILVER, fontName="Helvetica",
           leading=13, spaceBefore=2)
RULE_BODY = S("rule", fontSize=10, textColor=DARK_GREY, fontName="Helvetica",
               leading=16, spaceBefore=2, spaceAfter=2, leftIndent=12)
AGENT_TITLE = S("at", fontSize=16, textColor=WHITE, fontName="Helvetica-Bold",
                 leading=22, alignment=TA_LEFT)
AGENT_SUB   = S("asub", fontSize=10, textColor=MID_GREY, fontName="Helvetica-Oblique",
                 leading=14, alignment=TA_LEFT)

def agent_name_style(color):
    return S("an", fontSize=18, textColor=color, fontName="Helvetica-Bold",
              leading=24, spaceBefore=8, spaceAfter=4)

# ── Table style helpers ────────────────────────────────────────────────────────
def header_table_style(header_color=LIGHT_BLUE):
    return TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), header_color),
        ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 10),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,1), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT_GREY]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#B0BEC5")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
    ])

def agent_card_bg(color):
    return TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("TEXTCOLOR",  (0,0), (-1,-1), WHITE),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING",(0,0), (-1,-1), 14),
        ("RIGHTPADDING",(0,0),(-1,-1), 14),
        ("ROUNDEDCORNERS", [6]),
    ])

def hr(): return HRFlowable(width="100%", thickness=1.2,
                             color=GOLD, spaceAfter=8, spaceBefore=8)
def hr_light(): return HRFlowable(width="100%", thickness=0.5,
                                   color=colors.HexColor("#CFD8DC"),
                                   spaceAfter=6, spaceBefore=6)

# ── Build document ─────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=A4,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
        leftMargin=2.0*cm, rightMargin=2.0*cm,
    )
    story = []

    # ══════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ══════════════════════════════════════════════════════
    # Dark cover background via table
    cover_content = [
        [Paragraph("", COVER_TITLE)],
        [Paragraph("FIFTO", COVER_TITLE)],
        [Paragraph("Fusion Intraday Formula for Tactical Options", COVER_SUB)],
        [Spacer(1, 0.5*cm)],
        [Paragraph("NIFTY Weekly Options · Intraday Selling System", COVER_TAG)],
        [Paragraph("Five-Year Verified Performance · 2021–2026", COVER_TAG)],
        [Spacer(1, 1.2*cm)],
        [Paragraph("─── The Avengers of the Market ───", COVER_TAG)],
        [Spacer(1, 0.8*cm)],
        [Paragraph("7 Agents · 7 Zones · One Systematic Framework", COVER_TAG)],
        [Spacer(1, 2.0*cm)],
        [Paragraph("Confidential · For Authorized Recipients Only", SMALL)],
    ]
    cover_table = Table([[row[0]] for row in cover_content],
                         colWidths=[16*cm])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 30),
        ("RIGHTPADDING",  (0,0), (-1,-1), 30),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
    ]))
    story.append(cover_table)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE 2 — WHAT IS FIFTO
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("What is FIFTO?", H1))
    story.append(hr())
    story.append(Paragraph(
        "FIFTO is a rules-based intraday option selling system built on NIFTY weekly options. "
        "It identifies high-probability zones where the market is likely to reverse or stall, "
        "and sells option premium at those zones — capturing time decay and mean reversion "
        "with a disciplined exit framework.", BODY))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "The system combines <b>seven independent strategies</b> — each assigned to a Marvel "
        "Avenger character — that operate across different price zones on the NIFTY chart. "
        "Every agent has precise entry rules, fixed target, and a mechanical trailing stop-loss. "
        "Human judgement is <b>removed from the execution loop.</b>", BODY))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Five Core Principles", H2))
    principles = [
        ("1. Zone-First", "Never sell blindly. Every trade requires spot price to be at a defined zone (CPR, Camarilla, Fibonacci, or Pivot level)."),
        ("2. Premium Sell Only", "All trades are option SELL (short premium). We collect theta, not pay it. CE sell = bearish signal. PE sell = bullish reversal."),
        ("3. Mechanical Exit", "Target = 30% of entry premium. Trail SL activates automatically at -25%, -40%, -60% milestones. No manual intervention."),
        ("4. One Trade Per Day", "Maximum one position per day. No averaging, no doubling down, no revenge trading."),
        ("5. Capital Safety First", "Hard SL = 100% of entry premium (option doubles = immediate exit). Monthly loss months in 5 years: 3 out of 58."),
    ]
    for title, desc in principles:
        story.append(Paragraph(f"<b>{title}</b>: {desc}", RULE_BODY))
        story.append(Spacer(1, 0.15*cm))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE 3 — THE SELLING ZONES
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("The FIFTO Selling Zones", H1))
    story.append(hr())
    story.append(Paragraph(
        "FIFTO maps seven price zones on the NIFTY chart using standard technical levels "
        "computed from the <b>previous day's OHLC</b> data. All levels are calculated before market "
        "open — zero forward bias.", BODY))
    story.append(Spacer(1, 0.4*cm))

    zone_data = [
        ["Zone", "Level Formula", "Direction", "Agent"],
        ["CPR Zone\n(TC to PDH)", "TC = 2×Pivot − BC\nPDH = Previous Day High", "PE Sell\n(Bearish above)", "THOR"],
        ["CPR Breakdown\n(PDH to R1)", "Camarilla L3 = Close − Range×0.275", "PE Sell\n(Continuation)", "HULK"],
        ["R1 / R2\nResistance", "R1 = 2×Pivot − Low\nR2 = Pivot + Range", "PE Sell / CE Sell\n(Level rejection)", "IRON MAN"],
        ["PDL Zone\n(Previous Day Low)", "PDL = Previous Day Low", "CE Sell\n(Bearish breakdown)", "CAPTAIN"],
        ["TC / R1\nSweep Trap", "Price wicks above TC or R1\nthen snaps back below", "CE Sell\n(Reversal)", "SPIDER-MAN"],
        ["Fibonacci\nMean Reversion", "l_382 = PDH − Range×0.382\nl_618 = PDH − Range×0.618", "PE Sell\n(Reversion up)", "BLACK WIDOW"],
        ["Post-Target\nRe-entry", "60–75% pullback after\nfirst trade hits target", "Same as base\ntrade direction", "HAWKEYE"],
    ]
    zone_table = Table(zone_data, colWidths=[3.5*cm, 4.5*cm, 3.0*cm, 3.0*cm])
    zone_table.setStyle(header_table_style(DARK))
    story.append(zone_table)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Level Computation Reference", H3))
    level_ref = [
        ["Variable", "Formula", "Data Source"],
        ["Pivot (PP)",   "(H + L + C) / 3",                    "Previous day H/L/C"],
        ["BC",           "(H + L) / 2",                         "Previous day H/L"],
        ["TC",           "2 × PP − BC",                         "Derived"],
        ["R1",           "2 × PP − L",                          "Previous day L"],
        ["R2",           "PP + (H − L)",                        "Previous day H/L"],
        ["CAM L3",       "Close − Range × 0.275",               "Previous day Close + Range"],
        ["CAM H3",       "Close + Range × 0.275",               "Previous day Close + Range"],
        ["MRC l_618",    "PDH − Range × 0.618",                  "Previous day H/L"],
        ["MRC l_382",    "PDH − Range × 0.382",                  "Previous day H/L"],
        ["Futures Basis","Futures LTP − Spot LTP at 09:15",     "Live tick at open"],
        ["IB High/Low",  "Max/Min of NIFTY spot 09:15–09:45",   "Live ticks"],
    ]
    ref_table = Table(level_ref, colWidths=[3*cm, 5.5*cm, 5.5*cm])
    ref_table.setStyle(header_table_style(LIGHT_BLUE))
    story.append(ref_table)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGES 4–10 — THE SEVEN AGENTS
    # ══════════════════════════════════════════════════════
    agents = [
        {
            "name": "THOR",
            "subtitle": "The Zone Destroyer · v17a Strategy",
            "color": THOR_COLOR,
            "power": "Strikes from above — sells PE when price is elevated in the TC-to-PDH zone.",
            "zone": "TC to PDH / R1-to-R2 / within CPR",
            "opt": "PE Sell (bearish zones)",
            "entry": "Spot price enters the defined zone intraday. Signal candle confirms. Entry = next candle open + 2 seconds.",
            "lots": "Score-based: 1–3 lots (based on 7-feature conviction score)",
            "stats": [["Trades (5yr)", "WR", "Avg P&L/Trade", "Best Strategy"],
                      ["~220", "75.3%", "Rs. 1,950", "Most frequent agent"]],
            "rules": [
                "Zone must be defined from previous day data before market open.",
                "Spot price must enter zone (not just approach).",
                "No trade if IB (09:15–09:45) range is expanding beyond zone.",
                "Entry strictly at next candle open + 2 seconds after signal.",
                "Maximum 1 THOR trade per day.",
            ],
        },
        {
            "name": "HULK",
            "subtitle": "The Breakdown Hammer · cam_l3 Strategy",
            "color": HULK_COLOR,
            "power": "Smashes bearish zones — sells PE when price is in the PDH-to-R1 collapse channel.",
            "zone": "PDH to R1 (Camarilla L3 zone)",
            "opt": "PE Sell",
            "entry": "Price pulls up into Camarilla L3 zone. Signal candle shows rejection. Entry = next candle +2s.",
            "lots": "Score-based: 1–3 lots",
            "stats": [["Trades (5yr)", "WR", "Avg P&L/Trade", "Typical Exit"],
                      ["~55", "79.4%", "Rs. 2,480", "Target (73%)"]],
            "rules": [
                "CAM L3 = Close − Range × 0.275 (computed pre-market).",
                "Price must rally into CAM L3 zone, not gap up through it.",
                "Best combined with bearish CPR bias (open below BC).",
                "Score filter: at least 2 confluence features required.",
            ],
        },
        {
            "name": "IRON MAN",
            "subtitle": "The Precision Sniper · cam_h3 / iv2_r1 / iv2_r2",
            "color": IRON_COLOR,
            "power": "High-precision targeting — sells PE or CE at R1/R2/CAM H3 resistance levels.",
            "zone": "R1, R2, CAM H3 — upper resistance zones",
            "opt": "PE Sell (at R1/R2 resistance) · CE Sell (at pdl_to_bc breakdown)",
            "entry": "Price reaches R1 or R2 level. Candle signal confirms. Entry next candle +2s.",
            "lots": "Score-based: 1–3 lots",
            "stats": [["Trades (5yr)", "WR", "Avg P&L/Trade", "Best Zone"],
                      ["~75", "73.2%", "Rs. 2,100", "pdl_to_bc (highest WR)"]],
            "rules": [
                "R1 = 2×Pivot − Low (from previous day data).",
                "CAM H3 = Close + Range × 0.275.",
                "NOT used with cam_h3 + tc_to_pdh combination (structural mismatch — removed).",
                "Futures basis must be between -50 and +100 pts.",
            ],
        },
        {
            "name": "CAPTAIN",
            "subtitle": "The Reliable Soldier · iv2_pdl / iv2_r1 / iv2_r2",
            "color": CAP_COLOR,
            "power": "Never gives up — consistent level-based entries at PDL, R1, R2. Most reliable win rate.",
            "zone": "PDL, R1, R2 — exact key level entries",
            "opt": "CE Sell (at PDL breakdown) · PE Sell (at R1/R2)",
            "entry": "Spot price tags or tests key level. IV2 pattern confirms. Entry next candle +2s.",
            "lots": "1 lot always (no conviction score for iv2, score = -1)",
            "stats": [["Trades (5yr)", "WR", "Avg P&L/Trade", "Note"],
                      ["~55", "72.7%", "Rs. 842", "Always 1-lot, score not applicable"]],
            "rules": [
                "PDL = Previous day Low (pre-market calculation).",
                "Level must be within ±0.3% of spot at entry time.",
                "Entry between 09:16 and 15:15 only.",
                "No re-entry if CAPTAIN already traded today.",
            ],
        },
        {
            "name": "SPIDER-MAN",
            "subtitle": "The Web Trap · CRT (Candle Range Theory)",
            "color": SPIDER_COLOR,
            "power": "Sets the trap — price sweeps above TC or R1, lures bulls, then snaps back below. CE sell.",
            "zone": "TC or R1 — sweep and reverse pattern",
            "opt": "CE Sell (bearish reversal after sweep)",
            "entry": (
                "Step 1: 15-minute 3-candle pattern — C2 wicks above TC/R1, C3 closes back below.\n"
                "Step 2: 5-minute Heiken Ashi LTF confirmation — HA candle with upper wick + red body.\n"
                "Step 3: Entry = next 5M candle + 2 seconds."
            ),
            "lots": "1 lot",
            "stats": [["Trades (5yr)", "WR", "Avg P&L/Trade", "Active Days"],
                      ["136", "69.9%", "Rs. 551", "Blank days only (no base signal)"]],
            "rules": [
                "ONLY fires on blank days (days without base strategy signal).",
                "C3 of 15M pattern must close by 12:00 PM.",
                "If IB already expanded UP before entry → trade is SKIPPED (forward bias filter).",
                "Futures basis must be in range: −50 to +100 pts.",
                "LTF confirmation window: 30 minutes after C3 close.",
            ],
        },
        {
            "name": "BLACK WIDOW",
            "subtitle": "The Silent Reversal · MRC (Mean Reversion Concept)",
            "color": WIDOW_COLOR,
            "power": "Strikes from the shadows — PE sell when market bounces off the 38.2% Fibonacci level.",
            "zone": "l_382 = PDH − Range × 0.382 (Fibonacci mean reversion zone)",
            "opt": "PE Sell (bullish reversal at l_382) · <b>2 lots always</b> (WR 80.6% justifies double size)",
            "entry": (
                "5-minute Heiken Ashi green candle closes ABOVE l_382 level.\n"
                "Entry = next 5M candle + 2 seconds."
            ),
            "lots": "<b>2 lots</b> (approved based on 80.6% WR and low hard-SL rate of 4.7%)",
            "stats": [["Trades (5yr)", "WR", "Avg P&L/Trade (2-lot)", "Active Days"],
                      ["170", "80.6%", "Rs. 1,808", "Blank days only"]],
            "rules": [
                "ONLY fires on blank days (days without base strategy signal).",
                "Previous day range must be > 50 points (skip low-volatility days).",
                "HA candle: ha_close > l_382 AND ha_close > ha_open (green).",
                "Signal window: 09:15–12:00 PM only.",
                "MRC CE trades (below l_618) are EXCLUDED — net negative over 5 years.",
                "2 lots is fixed — no score-based adjustment for this agent.",
            ],
        },
        {
            "name": "HAWKEYE",
            "subtitle": "The Precision Re-entry · S4 Second Trade",
            "color": HAWK_COLOR,
            "power": "Never misses the second shot — re-enters after the first trade hits target, on a 60–75% pullback.",
            "zone": "Same as base strategy (THOR/HULK/IRON MAN/CAPTAIN) — re-entry after target",
            "opt": "Same direction as base trade (CE or PE)",
            "entry": (
                "Condition 1: Base trade (THOR/HULK/IRON MAN/CAPTAIN) hits 30% target before 13:30.\n"
                "Condition 2: Option premium then pulls back 60–75% of entry price.\n"
                "Entry = next candle after pullback is confirmed + 2 seconds."
            ),
            "lots": "1 lot",
            "stats": [["Trades (5yr)", "WR", "Avg P&L/Trade", "Trigger Rate"],
                      ["196", "71.4%", "Rs. 828", "~44% of base target days"]],
            "rules": [
                "Base trade must hit target BEFORE 13:30.",
                "Wait for option to pull back to 60–75% of original entry price.",
                "Same direction as original base trade.",
                "Only 1 HAWKEYE re-entry per day.",
                "EOD exit at 15:20 if not hit target.",
            ],
        },
    ]

    for agent in agents:
        color = agent["color"]

        # Agent header card
        header_data = [[
            Paragraph(f"⚡ {agent['name']}", AGENT_TITLE),
            Paragraph(agent["subtitle"], AGENT_SUB),
        ]]
        header_table = Table(header_data, colWidths=[5*cm, 11*cm])
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), color),
            ("TOPPADDING",    (0,0), (-1,-1), 12),
            ("BOTTOMPADDING", (0,0), (-1,-1), 12),
            ("LEFTPADDING",   (0,0), (-1,-1), 16),
            ("RIGHTPADDING",  (0,0), (-1,-1), 16),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(KeepTogether([
            header_table,
            Spacer(1, 0.3*cm),
        ]))

        # Power description
        story.append(Paragraph(f"<b>Power:</b> {agent['power']}", BODY))
        story.append(Spacer(1, 0.2*cm))

        # Details grid
        details = [
            ["Zone",  agent["zone"]],
            ["Trade", agent["opt"]],
            ["Entry", agent["entry"]],
            ["Lots",  agent["lots"]],
        ]
        det_table = Table(details, colWidths=[2.5*cm, 13.5*cm])
        det_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (0,-1), MID_GREY),
            ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("FONTNAME",    (1,0), (1,-1), "Helvetica"),
            ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#B0BEC5")),
            ("TOPPADDING",  (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ]))
        story.append(det_table)
        story.append(Spacer(1, 0.25*cm))

        # Stats
        stats_table = Table(agent["stats"],
                             colWidths=[3.5*cm, 3*cm, 4*cm, 5.5*cm])
        stats_table.setStyle(header_table_style(color))
        story.append(stats_table)
        story.append(Spacer(1, 0.25*cm))

        # Rules
        story.append(Paragraph("Rules & Filters:", H3))
        for i, rule in enumerate(agent["rules"]):
            story.append(Paragraph(f"  {i+1}. {rule}", RULE_BODY))
        story.append(Spacer(1, 0.2*cm))
        story.append(hr_light())
        story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE: UNIVERSAL ENTRY RULES
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Universal Entry Rules", H1))
    story.append(hr())

    entry_rules = [
        ("No Forward Bias",
         "All levels (CPR, Camarilla, Fibonacci, PDH/PDL) are computed from "
         "<b>previous day data</b> before market opens. No current-day future data is used."),
        ("Entry Timing",
         "Every entry is: <b>next candle open + 2 seconds</b>. This prevents entering "
         "on the signal candle itself. Window: 09:16 to 15:15."),
        ("IB Filter (CRT)",
         "For SPIDER-MAN: if IB (Initial Balance = 09:15–09:45 range) has <b>already expanded "
         "upward</b> before the entry time, the trade is skipped. IB expansion confirms bullish "
         "trend — selling CE into that is against the trend."),
        ("Futures Basis Filter (CRT)",
         "For SPIDER-MAN: futures basis must be between <b>−50 and +100 pts</b>. Extreme basis "
         "values indicate market dislocation — avoid selling in these conditions."),
        ("PE Basis Filter (Base)",
         "PE sells (THOR/HULK) are skipped when futures basis is <b>50–100 pts</b>. "
         "This range indicates strong bullish premium — PE sellers face adverse conditions."),
        ("Blank Day Logic",
         "SPIDER-MAN and BLACK WIDOW <b>only trade on blank days</b> — days when none of the "
         "base agents (THOR/HULK/IRON MAN/CAPTAIN) fire. On base days, only base + HAWKEYE trade."),
        ("Sequential Only",
         "Only <b>one active position</b> at a time. A new signal is only considered after "
         "the current trade exits. Exception: HAWKEYE fires after base exits with target."),
    ]
    for title, desc in entry_rules:
        story.append(Paragraph(f"<b>{title}</b>", H3))
        story.append(Paragraph(desc, BODY))
        story.append(Spacer(1, 0.1*cm))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE: UNIVERSAL EXIT RULES
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Universal Exit Rules", H1))
    story.append(hr())
    story.append(Paragraph(
        "All agents share the same exit framework. Once in a trade, the rules below "
        "apply mechanically — no manual overrides.", BODY))
    story.append(Spacer(1, 0.3*cm))

    exit_data = [
        ["Exit Type", "Trigger", "Typical P&L"],
        ["TARGET",     "Option price drops 30% from entry (ep × 0.70)",   "Rs. +1,500–6,000"],
        ["TRAIL SL\n(BE)",   "Option falls 25% from entry → SL moves to entry price",        "Rs. ~0"],
        ["TRAIL SL\n(Lock)", "Option falls 40% from entry → SL moves to 80% of entry",       "Rs. ~+300–1,500"],
        ["TRAIL SL\n(Max)",  "Option falls 60%+ from entry → SL trails at 95% of max move",  "Rs. ~+large"],
        ["HARD SL",    "Option price rises 100% from entry (doubles)",     "Rs. −4,000–17,000"],
        ["EOD EXIT",   "15:20:00 — force exit regardless of position",     "Rs. −/+"],
    ]
    exit_table = Table(exit_data, colWidths=[3.5*cm, 8.5*cm, 4*cm])
    exit_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), DARK),
        ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 10),
        ("BACKGROUND",  (0,1), (-1,1), colors.HexColor("#E8F5E9")),  # target
        ("BACKGROUND",  (0,2), (-1,2), LIGHT_GREY),
        ("BACKGROUND",  (0,3), (-1,3), LIGHT_GREY),
        ("BACKGROUND",  (0,4), (-1,4), LIGHT_GREY),
        ("BACKGROUND",  (0,5), (-1,5), colors.HexColor("#FFEBEE")),  # hard SL
        ("BACKGROUND",  (0,6), (-1,6), LIGHT_GREY),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,1), (-1,-1), 9),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#B0BEC5")),
        ("TOPPADDING",  (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
    ]))
    story.append(exit_table)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Trail SL Visual", H2))
    trail_desc = [
        ("Entry price = 100",     "Example: Sold CE @ Rs.100"),
        ("Target = Rs.70",        "Hit at −30% → exit with full profit"),
        ("Trail 1: Max −25%",     "Price falls to Rs.75 → SL moved to Rs.100 (breakeven)"),
        ("Trail 2: Max −40%",     "Price falls to Rs.60 → SL moved to Rs.80 (lock 20%)"),
        ("Trail 3: Max −60%",     "Price falls to Rs.40 → SL trails at Rs.42 (lock 58%)"),
        ("Hard SL: Rs.200",       "Price rises to Rs.200 → immediate exit, max loss"),
    ]
    for item, desc in trail_desc:
        story.append(Paragraph(
            f"<b>{item}</b>: {desc}", RULE_BODY))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "5-year exit breakdown: <b>Target 63.5%</b> · Trail SL 8.9% · EOD 20.7% · Hard SL 6.8%", BODY))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE: CONVICTION SCORING
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Conviction Scoring — Lot Sizing", H1))
    story.append(hr())
    story.append(Paragraph(
        "For base agents (THOR, HULK, IRON MAN, CAPTAIN), lot size is determined by "
        "a <b>7-feature conviction score</b> computed from previous-day data. "
        "Higher conviction = more lots = more profit when right.", BODY))
    story.append(Spacer(1, 0.3*cm))

    score_data = [
        ["Feature", "Condition for 1 Point", "Why It Matters"],
        ["VIX OK",           "India VIX today < 20-day MA of VIX",          "Low fear = premium sellers win"],
        ["CPR Trend Aligned","Prev close on correct side of CPR for direction","Prior day momentum aligned"],
        ["Consec Aligned",   "2 consecutive days close above/below CPR",    "Multi-day momentum confirmation"],
        ["CPR Gap Aligned",  "Today's open gaps in direction of trade",      "Gap = strong directional intent"],
        ["DTE Sweet Spot",   "Days to expiry = 3 to 5 days",                "Theta decay accelerates here"],
        ["CPR Narrow",       "CPR width between 0.10% and 0.20% of spot",   "Narrow CPR = trending day likely"],
        ["CPR Dir Aligned",  "CPR midpoint trending in trade direction 3 days","Macro CPR trend confirmation"],
    ]
    score_table = Table(score_data, colWidths=[3.5*cm, 5.5*cm, 7*cm])
    score_table.setStyle(header_table_style(DARK))
    story.append(score_table)
    story.append(Spacer(1, 0.4*cm))

    lot_data = [
        ["Score", "Lots", "Inside CPR?", "Final Lots", "Meaning"],
        ["0–1",  "1",    "No",          "1",          "Weak setup — minimum size"],
        ["0–1",  "1",    "Yes",         "1",          "Inside CPR reduces by 1 (min 1)"],
        ["2–3",  "2",    "No",          "2",          "Decent setup — standard size"],
        ["2–3",  "2",    "Yes",         "1",          "Inside CPR penalty applied"],
        ["4–7",  "3",    "No",          "3",          "High conviction — full size"],
        ["4–7",  "3",    "Yes",         "2",          "Inside CPR reduces slightly"],
        ["BLACK WIDOW PE", "2", "N/A",  "2",          "Fixed 2-lot regardless of score"],
        ["SPIDER-MAN CE",  "1", "N/A",  "1",          "Fixed 1-lot"],
        ["HAWKEYE",        "1", "N/A",  "1",          "Fixed 1-lot"],
    ]
    lot_table = Table(lot_data, colWidths=[3*cm, 1.8*cm, 2.5*cm, 2.5*cm, 6.2*cm])
    lot_table.setStyle(header_table_style(LIGHT_BLUE))
    story.append(lot_table)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE: PERFORMANCE SUMMARY
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("5-Year Performance Summary", H1))
    story.append(hr())

    # Top stats boxes
    top_stats = [
        ["TOTAL P&L", "Rs. 16,96,299"],
        ["WIN RATE",  "74.5%"],
        ["MAX DD",    "Rs. 46,134  (2.93%)"],
        ["COVERAGE",  "753 / 1155 days"],
    ]
    top_table = Table([
        [Paragraph(f"<b>{r[0]}</b>", S("ts", fontSize=9, textColor=SILVER, fontName="Helvetica-Bold",
                                        alignment=TA_CENTER, leading=13)),
         Paragraph(r[1], S("tv", fontSize=16, textColor=GOLD, fontName="Helvetica-Bold",
                             alignment=TA_CENTER, leading=20))]
        for r in top_stats
    ], colWidths=[8*cm, 8*cm], rowHeights=[1.5*cm]*4)
    top_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), DARK),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("GRID",        (0,0), (-1,-1), 1, GOLD),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
    ]))
    story.append(top_table)
    story.append(Spacer(1, 0.5*cm))

    # Year breakdown
    story.append(Paragraph("Year-wise Performance", H2))
    yr_data = [
        ["Year", "Trades", "Win Rate", "P&L (Rs.)", "Avg/Trade (Rs.)", "Trend"],
        ["2021", "164", "72.6%", "1,93,434",  "1,179", "Base Year"],
        ["2022", "207", "70.0%", "3,34,965",  "1,618", "+73% vs 2021"],
        ["2023", "202", "72.3%", "2,37,289",  "1,175", "Steady"],
        ["2024", "138", "76.8%", "2,97,083",  "2,153", "↑ Improving"],
        ["2025", "177", "81.4%", "4,23,023",  "2,390", "↑↑ Best WR"],
        ["2026*","61",  "77.0%", "2,10,506",  "3,451", "Partial year"],
        ["TOTAL","949", "74.5%", "16,96,299", "1,788", "5-year avg"],
    ]
    yr_table = Table(yr_data, colWidths=[1.5*cm, 2*cm, 2.2*cm, 3.5*cm, 3.5*cm, 3.3*cm])
    ts_yr = header_table_style(DARK)
    ts_yr.add("BACKGROUND", (0, 7), (-1, 7), LIGHT_BLUE)   # total row
    ts_yr.add("FONTNAME",   (0, 7), (-1, 7), "Helvetica-Bold")
    ts_yr.add("TEXTCOLOR",  (0, 7), (-1, 7), WHITE)
    yr_table.setStyle(ts_yr)
    story.append(yr_table)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("* 2026 is a partial year (January–April only).", SMALL))
    story.append(Spacer(1, 0.4*cm))

    # Agent breakdown
    story.append(Paragraph("Performance by Agent Group", H2))
    agent_perf = [
        ["Agent Group",       "Trades", "Win Rate", "Total P&L",  "Avg/Trade"],
        ["THOR + HULK +\nIRON MAN + CAPTAIN\n(Base Agents)", "447", "74.9%",
         "Rs. 11,51,716", "Rs. 2,577"],
        ["HAWKEYE\n(Re-entry)", "196", "71.4%", "Rs. 1,62,302", "Rs. 828"],
        ["SPIDER-MAN\n(CRT)", "136", "69.9%", "Rs. 74,935",  "Rs. 551"],
        ["BLACK WIDOW\n(MRC PE, 2-lot)", "170", "80.6%", "Rs. 3,07,346", "Rs. 1,808"],
        ["ALL AGENTS", "949", "74.5%", "Rs. 16,96,299", "Rs. 1,788"],
    ]
    ap_table = Table(agent_perf, colWidths=[4.5*cm, 2*cm, 2.2*cm, 3.8*cm, 3.5*cm])
    ts_ap = header_table_style(DARK)
    ts_ap.add("BACKGROUND", (0,5), (-1,5), LIGHT_BLUE)
    ts_ap.add("FONTNAME",   (0,5), (-1,5), "Helvetica-Bold")
    ts_ap.add("TEXTCOLOR",  (0,5), (-1,5), WHITE)
    ap_table.setStyle(ts_ap)
    story.append(ap_table)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE: RISK MANAGEMENT
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Risk Management", H1))
    story.append(hr())

    story.append(Paragraph("Capital Requirements", H2))
    cap_data = [
        ["Requirement", "Value", "Notes"],
        ["Minimum Capital",     "Rs. 5,00,000",  "For 1-lot base trades with adequate margin"],
        ["Margin per lot",      "Rs. ~50,000",   "Nifty ATM option sell margin (approx.)"],
        ["Max lots at once",    "3 lots",         "High conviction base trades only"],
        ["Daily risk (worst)",  "Rs. ~17,000",   "1 hard SL on a 3-lot base trade"],
        ["Monthly avg income",  "Rs. 29,247",    "Based on 58-month average"],
        ["Negative months",     "3 out of 58",   "5.2% of months — very low"],
        ["Worst single month",  "Rs. −24,000",   "2 hard SL in same month (rare)"],
        ["Best single month",   "Rs. 1,08,030",  "March 2026"],
    ]
    cap_table = Table(cap_data, colWidths=[5*cm, 4*cm, 7*cm])
    cap_table.setStyle(header_table_style(DARK_GREY))
    story.append(cap_table)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("What Controls Risk", H2))
    risk_controls = [
        ("Hard SL at 2×", "Option can never lose more than 100% of entry premium on any trade."),
        ("1 Trade Per Day", "A bad day only affects 1 trade. No compounding of losses."),
        ("Score Filter", "Low-conviction days trade 1 lot only — limiting exposure."),
        ("Basis Filter", "PE sells skipped when futures basis 50–100 → avoids adverse market structure."),
        ("IB Filter", "CRT CE skipped if IB already expanded up → avoids selling into confirmed uptrend."),
        ("EOD Exit", "No overnight holding. Fresh start every day."),
    ]
    for title, desc in risk_controls:
        story.append(Paragraph(f"<b>{title}:</b> {desc}", RULE_BODY))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Drawdown Profile", H2))
    dd_data = [
        ["Metric",             "Value"],
        ["Maximum Drawdown",   "Rs. 46,134 (2.93% of peak equity)"],
        ["Worst DD Period",    "Jul 15–24, 2024 — 4 trades, 2 hard SL in same week"],
        ["Recovery Time",      "Under 2 months in all DD periods"],
        ["DD > 5%",            "0 occurrences in 5 years"],
        ["Consecutive losses", "Max 5 in a row (rare, isolated to volatile weeks)"],
    ]
    dd_table = Table(dd_data, colWidths=[5*cm, 11*cm])
    dd_table.setStyle(header_table_style(DARK))
    story.append(dd_table)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE: PAPER TRADING PLAN
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Paper Trading Plan — Getting Started", H1))
    story.append(hr())
    story.append(Paragraph(
        "Before going live, FIFTO recommends a <b>30-day paper trading phase</b> to validate "
        "signal detection, entry timing, and exit execution against live market data. "
        "All results are logged to CSV — compare with backtest to confirm live-to-backtest alignment.", BODY))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Daily Routine", H2))
    routine = [
        ("08:55 AM", "Connect to broker API (Angel One SmartAPI). Fetch 45-day NIFTY daily OHLC. Compute all levels: CPR, Camarilla, MRC Fibonacci, Pivot levels, EMA20."),
        ("09:10 AM", "Print day plan to screen — TC, BC, R1, CAM L3/H3, MRC l_382/l_618, nearest expiry, score7 features."),
        ("09:15 AM", "Market opens. Start WebSocket for real-time NIFTY spot ticks. IB tracking begins."),
        ("09:16 AM", "Base agents active (THOR/HULK/IRON MAN/CAPTAIN). Monitor for zone entry signals."),
        ("09:46 AM", "IB confirmed. SPIDER-MAN and BLACK WIDOW scanners activate (blank days only)."),
        ("Signal fires", "Fetch ATM option price. Log paper entry. Start monitoring with trade manager."),
        ("During trade", "Trail SL updates automatically on each tick. No manual intervention needed."),
        ("15:20 PM", "Force EOD exit if still in trade. Log result."),
        ("15:30 PM", "Print daily summary. Compare with backtest expected P&L for the day."),
    ]
    rout_data = [[t, d] for t, d in routine]
    rout_table = Table(rout_data, colWidths=[3*cm, 13*cm])
    rout_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), MID_GREY),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#B0BEC5")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    story.append(rout_table)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Angel One Data Requirements", H2))
    api_data = [
        ["When",        "What",               "API Call",            "Used For"],
        ["08:55 AM",    "45-day NIFTY OHLC",  "getCandleData\n(ONE_DAY)", "CPR, EMA20, PDH/PDL"],
        ["09:15 AM",    "Futures LTP",         "ltpData()",           "Futures basis filter"],
        ["09:15–15:20", "NIFTY spot ticks",    "WebSocket token 26000","IB, 5M/15M candles"],
        ["On signal",   "Option LTP",          "ltpData(NFO)",        "Entry price"],
        ["On signal",   "Option ticks",        "WebSocket (NFO token)","SL/target monitoring"],
        ["Pre-market",  "Instrument master",   "ScripMaster JSON",    "NFO option tokens"],
    ]
    api_table = Table(api_data, colWidths=[2.5*cm, 4*cm, 3.5*cm, 6*cm])
    api_table.setStyle(header_table_style(DARK))
    story.append(api_table)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        "Paper trading logs are saved to <b>live/paper_trades.csv</b>. "
        "After 30 days, compare win rate, average P&L, and exit reason distribution with backtest. "
        "Expected alignment: WR within ±5%, avg P&L within ±20%.", BODY))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # FINAL PAGE — QUICK REFERENCE
    # ══════════════════════════════════════════════════════
    story.append(Paragraph("Quick Reference Card", H1))
    story.append(hr())

    qr_data = [
        ["Agent",         "Zone",            "Opt",  "Lots",  "WR"],
        ["THOR",          "TC–PDH / R1–R2",  "PE",   "1–3",   "75%"],
        ["HULK",          "PDH–R1 (CAM L3)", "PE",   "1–3",   "79%"],
        ["IRON MAN",      "R1/R2/CAM H3",    "PE/CE","1–3",   "73%"],
        ["CAPTAIN",       "PDL/R1/R2 (IV2)", "CE/PE","1",     "73%"],
        ["SPIDER-MAN",    "TC/R1 sweep trap","CE",   "1",     "70%"],
        ["BLACK WIDOW",   "l_382 Fibonacci",  "PE",   "2 ✓",  "81%"],
        ["HAWKEYE",       "Post-target re-entry","same","1",  "71%"],
    ]
    qr_table = Table(qr_data, colWidths=[3.5*cm, 4*cm, 1.8*cm, 2*cm, 2*cm])
    qr_table.setStyle(header_table_style(DARK))
    story.append(qr_table)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Exit Cheat Sheet", H2))
    ex_data = [
        ["Event",               "Action"],
        ["Option price = ep × 0.70",    "EXIT — Target hit (+30%)"],
        ["Max decline ≥ 25%",            "MOVE SL to entry (breakeven)"],
        ["Max decline ≥ 40%",            "MOVE SL to ep × 0.80 (lock 20%)"],
        ["Max decline ≥ 60%",            "TRAIL SL at 95% of max decline"],
        ["Option price = ep × 2.0",      "EXIT — Hard SL (−100%)"],
        ["Time ≥ 15:20:00",              "EXIT — EOD force exit"],
    ]
    ex_table = Table(ex_data, colWidths=[7*cm, 9*cm])
    ex_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), DARK),
        ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND",  (0,1), (-1,1), colors.HexColor("#E8F5E9")),
        ("BACKGROUND",  (0,5), (-1,5), colors.HexColor("#FFEBEE")),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#B0BEC5")),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(ex_table)
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph(
        "FIFTO v1.0 · Five-Year Backtest: 2021–2026 · 949 Trades · NIFTY Weekly Options · "
        "Verified bias-free, zero forward look-ahead.", SMALL))

    # ── Build ──────────────────────────────────────────────────────────────────
    doc.build(story)
    print(f"PDF created: {OUT_PATH}  ({os.path.getsize(OUT_PATH)//1024} KB)")


if __name__ == "__main__":
    build()
