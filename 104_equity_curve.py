"""
104_equity_curve.py — Final equity curve from parallel backtest (103)
======================================================================
Loads consolidated CSV from 103_parallel_backtest.py and plots:
  1. Base strategy alone
  2. CRT+MRC blank days (ATM 30%)
  3. Combined

Usage:
    python3 104_equity_curve.py
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, f'{os.path.expanduser("~")}/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import glob
import pandas as pd
from plot_util import send_custom_chart

OUT_DIR = 'data/20260430'

# ── Load parallel backtest results ────────────────────────────────────────────
csv_files = sorted(glob.glob('data/consolidated/103_crt_mrc_atm30_*.csv'))
if not csv_files:
    print("No consolidated CSV found. Run 103_parallel_backtest.py first.")
    sys.exit(1)
summary_csv = max(csv_files, key=os.path.getsize)
print(f"Loading: {summary_csv}")
df_bt = pd.read_csv(summary_csv)
df_bt['date'] = pd.to_datetime(df_bt['date'].astype(str), format='%Y%m%d')

# ── Load base strategy ────────────────────────────────────────────────────────
base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base['date'] = pd.to_datetime(base['date'].astype(str), format='mixed')
base_daily = base.groupby('date')['pnl_conv'].sum().reset_index()
base_daily.columns = ['date', 'base_pnl']

# ── Build daily blank P&L ─────────────────────────────────────────────────────
blank_daily = df_bt.groupby('date')['pnl'].sum().reset_index()
blank_daily.columns = ['date', 'blank_pnl']

# ── Merge on full date range ───────────────────────────────────────────────────
all_dates = pd.DataFrame({'date': sorted(set(base_daily['date']) | set(blank_daily['date']))})
m = all_dates.merge(base_daily, on='date', how='left').merge(blank_daily, on='date', how='left')
m['base_pnl']  = m['base_pnl'].fillna(0)
m['blank_pnl'] = m['blank_pnl'].fillna(0)
m['comb_pnl']  = m['base_pnl'] + m['blank_pnl']

m['base_eq']  = m['base_pnl'].cumsum()
m['blank_eq'] = m['blank_pnl'].cumsum()
m['comb_eq']  = m['comb_pnl'].cumsum()

# ── Stats ──────────────────────────────────────────────────────────────────────
total   = len(df_bt)
wins    = df_bt['win'].sum()
wr      = round(wins / total * 100, 2)
crt_df  = df_bt[df_bt['signal_type']=='CRT']
mrc_df  = df_bt[df_bt['signal_type']=='MRC']

base_dd  = (m['base_eq']  - m['base_eq'].cummax()).min()
blank_dd = (m['blank_eq'] - m['blank_eq'].cummax()).min()
comb_dd  = (m['comb_eq']  - m['comb_eq'].cummax()).min()

base_total  = round(m['base_pnl'].sum(), 0)
blank_total = round(m['blank_pnl'].sum(), 0)
comb_total  = round(m['comb_pnl'].sum(), 0)

print(f"\n{'='*65}")
print(f"  FINAL COMBINED EQUITY SUMMARY")
print(f"{'='*65}")
print(f"  Base strategy:  Rs.{base_total:>10,.0f}  ({len(base)} trades | Avg Rs.{base['pnl_conv'].mean():.0f})")
print(f"  CRT blank days: Rs.{crt_df['pnl'].sum():>10,.0f}  ({len(crt_df)}t | WR {crt_df['win'].mean()*100:.1f}% | Avg Rs.{crt_df['pnl'].mean():.0f})")
print(f"  MRC blank days: Rs.{mrc_df['pnl'].sum():>10,.0f}  ({len(mrc_df)}t | WR {mrc_df['win'].mean()*100:.1f}% | Avg Rs.{mrc_df['pnl'].mean():.0f})")
print(f"  Combined total: Rs.{comb_total:>10,.0f}")
print(f"\n  Max drawdown — Base: Rs.{base_dd:,.0f}  | Blank: Rs.{blank_dd:,.0f}  | Combined: Rs.{comb_dd:,.0f}")

profit_factor_blank = round(df_bt[df_bt['pnl']>0]['pnl'].sum() /
                             abs(df_bt[df_bt['pnl']<0]['pnl'].sum()), 2) if (df_bt['pnl']<0).any() else 0
print(f"\n  Blank strategy:  {total}t | WR {wr}% | Avg Rs.{round(blank_total/total,0):,.0f} | PF {profit_factor_blank}")

exits = df_bt['exit_reason'].value_counts()
print(f"  Exit breakdown: " + " | ".join(f"{r}: {c}" for r, c in exits.items()))

# ── Build tvJson ───────────────────────────────────────────────────────────────
def eq_pts(series, dates):
    return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
            for d, v in zip(dates, series) if pd.notna(v)]

dates = m['date']
comb_dd_series = m['comb_eq'] - m['comb_eq'].cummax()

tv_json = {
    "isTvFormat": False,
    "candlestick": [],
    "volume": [],
    "lines": [
        {
            "id": "combined",
            "label": f"Combined (Base + CRT/MRC ATM 30%) — Rs.{comb_total:,.0f}",
            "color": "#26a69a",
            "data": eq_pts(m['comb_eq'], dates),
            "seriesType": "line",
        },
        {
            "id": "base",
            "label": f"Base Strategy — Rs.{base_total:,.0f}",
            "color": "#0ea5e9",
            "data": eq_pts(m['base_eq'], dates),
            "seriesType": "line",
        },
        {
            "id": "blank",
            "label": f"CRT+MRC Blank Days ATM 30% — Rs.{blank_total:,.0f}",
            "color": "#f59e0b",
            "data": eq_pts(m['blank_eq'], dates),
            "seriesType": "line",
        },
        {
            "id": "comb_dd",
            "label": f"Combined Drawdown (max Rs.{comb_dd:,.0f})",
            "color": "#ef5350",
            "data": eq_pts(comb_dd_series, dates),
            "seriesType": "baseline",
            "baseValue": 0,
            "isNewPane": True,
        },
    ]
}

send_custom_chart("104_equity_curve", tv_json,
                  title=f"Final Equity — Base + CRT/MRC Blank Days (ATM 30% target) | {total}t | WR {wr}% | Combined Rs.{comb_total:,.0f}")

print("\nDone.")
