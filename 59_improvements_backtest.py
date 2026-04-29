"""
59_improvements_backtest.py
Test 5 strategy improvements on v17a trades:
  1. India VIX filter    — skip when VIX > 20-day MA
  2. Narrow CPR filter   — skip when CPR width > 0.3% of spot
  3. EP cap              — 3-lot only when 80 <= EP <= 200
  4. Expiry day window   — Thursday entry only before 09:30
  5. Cam confluence      — bonus trades on cam_l3/cam_h3 days
"""
import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
from plot_util import send_custom_chart

DATA_ROOT = os.environ['INTER_SERVER_DATA_PATH']
CSV_PATH  = 'data/56_combined_trades.csv'

def r2(v): return round(float(v), 2)

# ─────────────────────────────────────────────
# Load base trades
# ─────────────────────────────────────────────
df_all = pd.read_csv(CSV_PATH, parse_dates=['date'])
v17a   = df_all[df_all['strategy'] == 'v17a'].copy().sort_values('date').reset_index(drop=True)
print(f"Base v17a: {len(v17a)} trades | WR {(v17a.pnl>0).mean()*100:.1f}% | P&L Rs {v17a.pnl.sum():,.0f}")

# ─────────────────────────────────────────────
# Helper: load daily VIX close
# ─────────────────────────────────────────────
def load_daily_vix():
    from my_util import list_trading_dates
    dates = list_trading_dates()
    result = {}
    for d in dates:
        path = os.path.join(DATA_ROOT, d, 'INDIAVIX.csv')
        if not os.path.exists(path):
            continue
        try:
            tks = pd.read_csv(path, header=None, names=['date','time','price','vol','oi'])
            result[d] = float(tks['price'].iloc[-1])
        except Exception:
            continue
    vix_s = pd.Series(result)
    vix_s.index = pd.to_datetime(vix_s.index, format='%Y%m%d')
    return vix_s.sort_index()

# ─────────────────────────────────────────────
# Helper: load daily spot open + CPR width
# ─────────────────────────────────────────────
def load_daily_spot_and_cpr():
    from my_util import list_trading_dates, load_spot_data
    dates = list_trading_dates()
    ohlc_map = {}
    for d in dates:
        tks = load_spot_data(d, 'NIFTY')
        if tks is None or tks.empty:
            continue
        ohlc_map[d] = (float(tks['price'].max()), float(tks['price'].min()), float(tks['price'].iloc[-1]), float(tks['price'].iloc[0]))
    # compute CPR width for each date (using prev day H/L/C)
    cpr_width = {}
    spot_open = {}
    for i, d in enumerate(dates):
        if i == 0 or d not in ohlc_map:
            continue
        prev = dates[i-1]
        if prev not in ohlc_map:
            continue
        ph, pl, pc, _ = ohlc_map[prev]
        pp = (ph + pl + pc) / 3
        bc = (ph + pl) / 2
        tc = 2 * pp - bc
        today_open = ohlc_map[d][3]
        width_pct = abs(tc - bc) / today_open * 100
        cpr_width[d] = r2(width_pct)
        spot_open[d] = today_open
    return cpr_width, spot_open

# ─────────────────────────────────────────────
# Load all data
# ─────────────────────────────────────────────
print("\nLoading VIX data...")
vix_daily = load_daily_vix()
vix_20ma  = vix_daily.rolling(20).mean()

print("Loading CPR width data...")
cpr_width_map, spot_open_map = load_daily_spot_and_cpr()

# ─────────────────────────────────────────────
# Attach metadata to v17a
# ─────────────────────────────────────────────
def get_vix(dt):
    try:
        return float(vix_daily.loc[dt])
    except Exception:
        return np.nan

def get_vix_ma(dt):
    try:
        return float(vix_20ma.loc[dt])
    except Exception:
        return np.nan

v17a['date_dt']   = pd.to_datetime(v17a['date'])
v17a['date_str']  = v17a['date'].dt.strftime('%Y%m%d')
v17a['vix']       = v17a['date_dt'].map(get_vix)
v17a['vix_20ma']  = v17a['date_dt'].map(get_vix_ma)
v17a['cpr_width'] = v17a['date_str'].map(cpr_width_map)
v17a['weekday']   = v17a['date_dt'].dt.dayofweek   # 0=Mon … 3=Thu … 4=Fri
v17a['entry_hhmm']= v17a['entry_time'].str[:5]

# DTE proxy (Mon=3, Tue=2, Wed=1, Thu=0)
dte_map = {0:3, 1:2, 2:1, 3:0, 4:6}
v17a['dte'] = v17a['weekday'].map(dte_map)

# 3-lot flag: DTE>=3 AND EP>80 (current rule)
v17a['is_3lot'] = (v17a['dte'] >= 3) & (v17a['ep'] > 80)

# ─────────────────────────────────────────────
# Stats helper
# ─────────────────────────────────────────────
def stats(mask, label):
    sub  = v17a[mask]
    skip = v17a[~mask]
    n    = len(sub)
    if n == 0:
        print(f"  {label:<40} 0 trades — all filtered out")
        return None
    wr   = r2((sub.pnl > 0).mean() * 100)
    pnl  = r2(sub.pnl.sum())
    avg  = r2(sub.pnl.mean())
    skip_wr = r2((skip.pnl > 0).mean() * 100) if len(skip) > 0 else 0
    print(f"  {label:<40} {n:>3} trades | WR {wr:>5.1f}% | P&L Rs {pnl:>10,.0f} | Skipped {len(skip):>3} (WR {skip_wr:.1f}%)")
    return sub

# ─────────────────────────────────────────────
# FILTER 1: India VIX < 20MA
# ─────────────────────────────────────────────
print("\n" + "=" * 75)
print("  FILTER 1: India VIX < 20-day MA (skip high-volatility days)")
print("=" * 75)
f1_mask = (v17a['vix'] <= v17a['vix_20ma']) | v17a['vix'].isna()
stats(pd.Series([True]*len(v17a)), "Baseline (no filter)")
f1_sub = stats(f1_mask, "VIX <= 20MA (trade)")
stats(~f1_mask & v17a['vix'].notna(), "VIX > 20MA (skipped)")

# VIX threshold sensitivity
print("\n  VIX level sensitivity:")
for thresh in [12, 15, 18, 20]:
    mask = v17a['vix'] <= thresh
    stats(mask, f"  VIX <= {thresh}")

# ─────────────────────────────────────────────
# FILTER 2: Narrow CPR (width < threshold)
# ─────────────────────────────────────────────
print("\n" + "=" * 75)
print("  FILTER 2: Narrow CPR filter (CPR width % of spot)")
print("=" * 75)
for thresh in [0.15, 0.20, 0.25, 0.30, 0.40]:
    mask = v17a['cpr_width'] <= thresh
    stats(mask, f"  CPR width <= {thresh:.2f}%")

# Best CPR threshold
f2_mask = v17a['cpr_width'] <= 0.25
f2_sub  = stats(f2_mask, "CPR width <= 0.25% (selected)")

# ─────────────────────────────────────────────
# FILTER 3: EP cap (3-lot quality)
# ─────────────────────────────────────────────
print("\n" + "=" * 75)
print("  FILTER 3: EP cap — 3-lot only in 80–200 sweet spot")
print("=" * 75)
print("  Current 3-lot rule: DTE>=3 AND EP>80")
for ep_max in [120, 150, 180, 200, 250]:
    # Keep as 3-lot only if EP in [80, ep_max], else 1-lot (divide pnl by 3)
    adj_pnl = v17a.apply(lambda r: r['pnl'] if not r['is_3lot'] or r['ep'] <= ep_max
                         else r['pnl'] / 3, axis=1)
    orig_3lot_oor = v17a[v17a['is_3lot'] & (v17a['ep'] > ep_max)]
    n_downgraded  = len(orig_3lot_oor)
    tot = r2(adj_pnl.sum())
    wr  = r2((adj_pnl > 0).mean() * 100)
    print(f"  EP cap <= {ep_max}: total P&L Rs {tot:>10,.0f} | WR {wr:.1f}% | downgraded {n_downgraded} trades")

# ─────────────────────────────────────────────
# FILTER 4: Expiry day (Thursday) time window
# ─────────────────────────────────────────────
print("\n" + "=" * 75)
print("  FILTER 4: Thursday expiry — restrict entry window")
print("=" * 75)
thu = v17a[v17a['weekday'] == 3]
non_thu = v17a[v17a['weekday'] != 3]
print(f"  Non-Thursday ({len(non_thu)} trades): WR {(non_thu.pnl>0).mean()*100:.1f}% | P&L Rs {non_thu.pnl.sum():,.0f}")
print(f"  All Thursday ({len(thu)} trades):     WR {(thu.pnl>0).mean()*100:.1f}% | P&L Rs {thu.pnl.sum():,.0f}")
for cutoff in ['09:20', '09:25', '09:30', '09:45', '10:00']:
    t_keep = thu[thu['entry_hhmm'] <= cutoff]
    t_skip = thu[thu['entry_hhmm'] > cutoff]
    print(f"    Thu entry <= {cutoff}: keep {len(t_keep):>2} trades WR {(t_keep.pnl>0).mean()*100 if len(t_keep)>0 else 0:.1f}% P&L Rs {t_keep.pnl.sum():>8,.0f}  |  skip {len(t_skip):>2} trades WR {(t_skip.pnl>0).mean()*100 if len(t_skip)>0 else 0:.1f}%")

# ─────────────────────────────────────────────
# FILTER 5: Camarilla confluence
# ─────────────────────────────────────────────
print("\n" + "=" * 75)
print("  FILTER 5: Camarilla confluence (cam + v17a same day, same direction)")
print("=" * 75)
cam = df_all[df_all['strategy'].isin(['cam_l3','cam_h3'])].copy()
cam_dates_bull = set(cam[cam['strategy']=='cam_l3']['date'].astype(str))
cam_dates_bear = set(cam[cam['strategy']=='cam_h3']['date'].astype(str))
v17a['date_s'] = v17a['date'].astype(str)
# Confluence: v17a bull + cam_l3 on same date, or v17a bear + cam_h3
conf_mask = (
    ((v17a['bias']=='bull') & (v17a['date_s'].isin(cam_dates_bull))) |
    ((v17a['bias']=='bear') & (v17a['date_s'].isin(cam_dates_bear)))
)
print(f"  v17a + Cam confluence: {conf_mask.sum()} trades")
stats(conf_mask, "Confluence trades")
stats(~conf_mask, "Non-confluence trades")

# ─────────────────────────────────────────────
# COMBINED: VIX + narrow CPR
# ─────────────────────────────────────────────
print("\n" + "=" * 75)
print("  COMBINED FILTERS")
print("=" * 75)
combined_mask = f1_mask & f2_mask
stats(combined_mask, "VIX<=20MA + CPR<=0.25%")

vix_only = stats(f1_mask, "VIX<=20MA only")
cpr_only = stats(f2_mask, "CPR<=0.25% only")
both     = stats(combined_mask, "Both combined")

# ─────────────────────────────────────────────
# CHART: Equity curve comparison (baseline vs VIX filter vs CPR filter vs both)
# ─────────────────────────────────────────────
def equity_line(sub, label, color):
    s = sub.sort_values('date').reset_index(drop=True)
    eq = s['pnl'].cumsum()
    return {
        'id': label.lower().replace(' ','_').replace('+',''),
        'label': label,
        'color': color,
        'data': [{'time': int(pd.Timestamp(r['date']).timestamp()), 'value': round(eq[i], 2)}
                 for i, (_, r) in enumerate(s.iterrows())],
    }

lines = [
    equity_line(v17a,              'Baseline',       '#8b949e'),
    equity_line(v17a[f1_mask],     'VIX≤20MA',       '#26a69a'),
    equity_line(v17a[f2_mask],     'CPR≤0.25%',      '#4BC0C0'),
    equity_line(v17a[combined_mask],'VIX+CPR',       '#00C853'),
]

tv_eq = {'lines': lines, 'candlestick': [], 'volume': [], 'isTvFormat': False}
send_custom_chart('improvements_equity', tv_eq,
                  title='Strategy Improvements — Equity Curve Comparison (v17a)')
print("\n📊 Equity comparison chart sent")

# ─────────────────────────────────────────────
# CHART: WR comparison bar chart (all filters)
# ─────────────────────────────────────────────
filters = [
    ('Baseline',       v17a),
    ('VIX≤20MA',       v17a[f1_mask]),
    ('CPR≤0.25%',      v17a[f2_mask]),
    ('VIX+CPR',        v17a[combined_mask]),
    ('Thu≤09:30',      pd.concat([non_thu, thu[thu['entry_hhmm']<='09:30']])),
    ('Confluence',     v17a[conf_mask]),
]
ts_base = int(pd.Timestamp('2021-01-01').timestamp())
bar_wr  = []
bar_pnl = []
for i, (name, sub) in enumerate(filters):
    if len(sub) == 0:
        continue
    wr_val  = round((sub.pnl > 0).mean() * 100, 2)
    pnl_val = round(sub.pnl.sum(), 2)
    t = ts_base + i * 86400 * 180
    bar_wr.append({'time': t, 'value': wr_val,
                   'color': '#26a69a' if wr_val >= 72 else '#f59e0b',
                   'label': name})
    bar_pnl.append({'time': t, 'value': pnl_val,
                    'color': '#26a69a' if pnl_val >= 0 else '#ef5350',
                    'label': name})

tv_wr = {'lines': [{'id':'filter_wr','label':'Win Rate %','seriesType':'bar',
                    'data': bar_wr, 'xLabels': [b['label'] for b in bar_wr]}],
         'candlestick':[],'volume':[],'isTvFormat':False}
send_custom_chart('filter_wr', tv_wr, title='Filter Win Rate Comparison (v17a)')
print("📊 WR comparison chart sent")

tv_pnl = {'lines': [{'id':'filter_pnl','label':'Total P&L (Rs)','seriesType':'bar',
                     'data': bar_pnl, 'xLabels': [b['label'] for b in bar_pnl]}],
          'candlestick':[],'volume':[],'isTvFormat':False}
send_custom_chart('filter_pnl', tv_pnl, title='Filter Total P&L Comparison (v17a)')
print("📊 P&L comparison chart sent")

print("\nDone!")
