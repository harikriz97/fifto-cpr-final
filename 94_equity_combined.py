"""
94_equity_combined.py — Combined equity curve: Base strategy + CRT Approach D (blank days)
===========================================================================================
Shows 3 equity lines:
  1. Base strategy alone (75_live_simulation.csv)
  2. CRT Approach D blank days alone (91_crt_ltf_D.csv, is_blank=True)
  3. Combined (base + CRT on blank days)
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, f'{os.path.expanduser("~")}/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from plot_util import send_custom_chart

OUT_DIR = 'data/20260430'

# ── Load data ─────────────────────────────────────────────────────────────────
base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base['date'] = pd.to_datetime(base['date'].astype(str), format='mixed')
base_daily = base.groupby('date')['pnl_conv'].sum().reset_index()
base_daily.columns = ['date', 'pnl']

crt = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt = crt[crt['is_blank'] == True].copy()
crt['date'] = pd.to_datetime(crt['date'].astype(str), format='%Y%m%d')
crt_daily = crt.groupby('date')['pnl_65'].sum().reset_index()
crt_daily.columns = ['date', 'pnl']

# ── Build daily P&L on full date range ────────────────────────────────────────
all_dates = pd.DataFrame({'date': sorted(set(base_daily['date']) | set(crt_daily['date']))})

base_eq = all_dates.merge(base_daily.rename(columns={'pnl': 'base_pnl'}), on='date', how='left').fillna(0)
crt_eq  = all_dates.merge(crt_daily.rename(columns={'pnl': 'crt_pnl'}),  on='date', how='left').fillna(0)

merged = base_eq.copy()
merged['crt_pnl']      = crt_eq['crt_pnl'].values
merged['combined_pnl'] = merged['base_pnl'] + merged['crt_pnl']

merged['base_eq']     = merged['base_pnl'].cumsum()
merged['crt_eq']      = merged['crt_pnl'].cumsum()
merged['combined_eq'] = merged['combined_pnl'].cumsum()

# ── Stats ──────────────────────────────────────────────────────────────────────
print(f"Base strategy:    Rs.{merged['base_pnl'].sum():>10,.0f}  |  final equity Rs.{merged['base_eq'].iloc[-1]:,.0f}")
print(f"CRT blank days:   Rs.{merged['crt_pnl'].sum():>10,.0f}  |  final equity Rs.{merged['crt_eq'].iloc[-1]:,.0f}")
print(f"Combined:         Rs.{merged['combined_pnl'].sum():>10,.0f}  |  final equity Rs.{merged['combined_eq'].iloc[-1]:,.0f}")

base_dd  = (merged['base_eq']     - merged['base_eq'].cummax()).min()
crt_dd   = (merged['crt_eq']      - merged['crt_eq'].cummax()).min()
comb_dd  = (merged['combined_eq'] - merged['combined_eq'].cummax()).min()
print(f"\nMax drawdown — Base: Rs.{base_dd:,.0f}  |  CRT: Rs.{crt_dd:,.0f}  |  Combined: Rs.{comb_dd:,.0f}")

# ── Build tvJson ───────────────────────────────────────────────────────────────
def eq_points(series, dates):
    return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
            for d, v in zip(dates, series) if pd.notna(v)]

def dd_series(eq):
    dd = eq - eq.cummax()
    return dd

dates = merged['date']

tv_json = {
    "isTvFormat": False,
    "candlestick": [],
    "volume": [],
    "lines": [
        {
            "id": "combined",
            "label": "Combined (Base + CRT)",
            "color": "#26a69a",
            "data": eq_points(merged['combined_eq'], dates),
            "seriesType": "line",
        },
        {
            "id": "base",
            "label": "Base Strategy",
            "color": "#0ea5e9",
            "data": eq_points(merged['base_eq'], dates),
            "seriesType": "line",
        },
        {
            "id": "crt",
            "label": "CRT Approach D (blank days)",
            "color": "#f59e0b",
            "data": eq_points(merged['crt_eq'], dates),
            "seriesType": "line",
        },
        {
            "id": "combined_dd",
            "label": "Combined Drawdown",
            "color": "#ef5350",
            "data": eq_points(dd_series(merged['combined_eq']), dates),
            "seriesType": "baseline",
            "baseValue": 0,
            "isNewPane": True,
        },
    ]
}

send_custom_chart("94_equity_combined", tv_json,
                  title="Equity Curve — Base Strategy + CRT Approach D (blank days)")

print("\nDone.")
