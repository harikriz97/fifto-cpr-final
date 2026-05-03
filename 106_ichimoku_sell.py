"""
106_ichimoku_sell.py — Ichimoku Cloud directional selling on blank days
=======================================================================
Signal: Daily Nifty Ichimoku cloud position at day open
  Price above cloud → Sell ATM PE
  Price below cloud → Sell ATM CE
  Price inside cloud → Skip

Ichimoku components (daily):
  Tenkan  = (9H + 9L) / 2
  Kijun   = (26H + 26L) / 2
  Span A  = (Tenkan[D-26] + Kijun[D-26]) / 2  ← cloud on current bar
  Span B  = (52H + 52L) / 2 shifted 26 forward ← cloud on current bar
  cloud_top    = max(Span A, Span B)
  cloud_bottom = min(Span A, Span B)

Entry: 10:00:02, ATM strike, 30% target, trailing SL
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, f'{os.path.expanduser("~")}/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates
from plot_util import send_custom_chart

OUT_DIR    = 'data/20260430'
LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
EOD_EXIT   = '15:20:00'
ENTRY_TIME = '10:00:02'
TGT_PCT    = 0.30
YEARS      = 5

def r2(v): return round(float(v), 2)
def get_atm(spot): return int(round(spot / STRIKE_INT) * STRIKE_INT)

def simulate_sell(date_str, instrument, tgt_pct=TGT_PCT):
    tks = load_tick_data(date_str, instrument, ENTRY_TIME)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= ENTRY_TIME].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - tgt_pct))
    hsl = r2(ep * (1 + 1.00)); sl = hsl; md = 0.0
    ps = tks['price'].values; ts = tks['time'].values
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            return r2((ep - p) * LOT_SIZE * SCALE), 'eod', r2(ep), r2(p)
        d = (ep - p) / ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep - p) * LOT_SIZE * SCALE), 'target', r2(ep), r2(p)
        if p >= sl:  return r2((ep - p) * LOT_SIZE * SCALE), 'lockin_sl' if sl < hsl else 'hard_sl', r2(ep), r2(p)
    return r2((ep - ps[-1]) * LOT_SIZE * SCALE), 'eod', r2(ep), r2(ps[-1])

# ── Build daily OHLC for all dates ────────────────────────────────────────────
print("Building daily OHLC (need ~100+ days seed for Ichimoku)...")
t0 = time.time()
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
# Need 52+26=78 days seed before first signal
seed_idx  = max(0, all_dates.index(dates_5yr[0]) - 100)
scan_dates = all_dates[seed_idx:]

rows = []
for d in scan_dates:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) < 2: continue
    rows.append({
        'date': d,
        'o': day.iloc[0]['price'],
        'h': day['price'].max(),
        'l': day['price'].min(),
        'c': day.iloc[-1]['price'],
    })

df_daily = pd.DataFrame(rows).reset_index(drop=True)
print(f"  {len(df_daily)} daily candles loaded in {time.time()-t0:.0f}s")

# ── Compute Ichimoku ──────────────────────────────────────────────────────────
def rolling_mid(h, l, n):
    return ((h.rolling(n).max() + l.rolling(n).min()) / 2).round(2)

df_daily['tenkan'] = rolling_mid(df_daily['h'], df_daily['l'], 9)
df_daily['kijun']  = rolling_mid(df_daily['h'], df_daily['l'], 26)

# Span A and B on current bar = what was projected 26 bars forward from D-26
df_daily['span_a'] = ((df_daily['tenkan'].shift(26) + df_daily['kijun'].shift(26)) / 2).round(2)
df_daily['span_b'] = rolling_mid(df_daily['h'].shift(26), df_daily['l'].shift(26), 52)

df_daily['cloud_top']    = df_daily[['span_a','span_b']].max(axis=1)
df_daily['cloud_bottom'] = df_daily[['span_a','span_b']].min(axis=1)
df_daily['cloud_color']  = (df_daily['span_a'] >= df_daily['span_b']).map({True:'green', False:'red'})

# TK cross direction
df_daily['tk_bull'] = df_daily['tenkan'] > df_daily['kijun']

# Signal based on open vs cloud
def cloud_signal(row):
    if pd.isna(row['cloud_top']) or pd.isna(row['cloud_bottom']): return None
    if row['o'] > row['cloud_top']:    return 'PE'   # above cloud → bullish → sell PE
    if row['o'] < row['cloud_bottom']: return 'CE'   # below cloud → bearish → sell CE
    return None  # inside cloud → skip

df_daily['signal'] = df_daily.apply(cloud_signal, axis=1)
ichimoku_map = dict(zip(df_daily['date'], zip(
    df_daily['signal'], df_daily['cloud_top'], df_daily['cloud_bottom'],
    df_daily['tk_bull'], df_daily['cloud_color']
)))

# ── Load blank days ────────────────────────────────────────────────────────────
print("\nLoading blank days...")
# Full blank day list = CRT blank + remaining blank
crt_raw   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_blank = set(crt_raw[crt_raw['is_blank']==True]['date'].astype(str))

blank_remaining = pd.read_csv(f'{OUT_DIR}/95_blank_remaining.csv')
blank_remaining_set = set(blank_remaining['date'].astype(str))

all_blank = crt_blank | blank_remaining_set
blank_days = sorted([d for d in dates_5yr if d in all_blank])
print(f"  Total blank days: {len(blank_days)}")

# ── Run backtest ───────────────────────────────────────────────────────────────
print(f"\nRunning Ichimoku backtest on {len(blank_days)} blank days...")
t0 = time.time()
trades = []
skipped_inside = 0
skipped_nodata = 0

for dstr in blank_days:
    info = ichimoku_map.get(dstr)
    if info is None or info[0] is None:
        skipped_inside += 1
        continue
    signal, cloud_top, cloud_bottom, tk_bull, cloud_color = info

    spot = load_spot_data(dstr, 'NIFTY')
    if spot is None: skipped_nodata += 1; continue
    spot_at = spot[spot['time'] >= '10:00:00']
    if spot_at.empty: skipped_nodata += 1; continue
    spot_ref = spot_at.iloc[0]['price']

    expiries = list_expiry_dates(dstr, index_name='NIFTY')
    if not expiries: skipped_nodata += 1; continue
    strike = get_atm(spot_ref)
    instr  = f'NIFTY{expiries[0]}{strike}{signal}'

    res = simulate_sell(dstr, instr)
    if res is None: skipped_nodata += 1; continue
    pnl, reason, ep, xp = res

    trades.append({
        'date': dstr, 'signal': signal, 'strike': strike,
        'cloud_color': cloud_color, 'tk_bull': tk_bull,
        'cloud_top': cloud_top, 'cloud_bottom': cloud_bottom,
        'ep': ep, 'xp': xp, 'pnl': pnl,
        'win': pnl > 0, 'exit_reason': reason,
    })

elapsed = time.time() - t0
df_t = pd.DataFrame(trades)

# ── Stats ──────────────────────────────────────────────────────────────────────
total = len(df_t)
wins  = df_t['win'].sum()
wr    = round(wins / total * 100, 2)
total_pnl = round(df_t['pnl'].sum(), 2)
avg_pnl   = round(df_t['pnl'].mean(), 2)

ce_df = df_t[df_t['signal']=='CE']
pe_df = df_t[df_t['signal']=='PE']

print(f"\n{'='*65}")
print(f"  ICHIMOKU CLOUD SELLING — BLANK DAYS (ATM 30%)")
print(f"{'='*65}")
print(f"  Total: {total}t | WR: {wr}% | P&L: Rs.{total_pnl:,.0f} | Avg: Rs.{avg_pnl:,.0f}")
print(f"  CE sell: {len(ce_df)}t | WR {ce_df['win'].mean()*100:.1f}% | "
      f"Rs.{ce_df['pnl'].sum():,.0f} | Avg Rs.{ce_df['pnl'].mean():.0f}")
print(f"  PE sell: {len(pe_df)}t | WR {pe_df['win'].mean()*100:.1f}% | "
      f"Rs.{pe_df['pnl'].sum():,.0f} | Avg Rs.{pe_df['pnl'].mean():.0f}")
print(f"  Skipped (inside cloud): {skipped_inside} | No data: {skipped_nodata}")
print(f"  Time: {elapsed:.0f}s")

exits = df_t['exit_reason'].value_counts()
print(f"\n  Exit breakdown:")
for r, c in exits.items():
    print(f"    {r:<12}: {c} ({round(c/total*100,1)}%)")

# Cloud color breakdown
print(f"\n  Cloud color breakdown:")
for color, g in df_t.groupby('cloud_color'):
    print(f"    {color}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
          f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

# TK alignment breakdown
print(f"\n  TK aligned (signal matches TK direction):")
tk_aligned = df_t[
    ((df_t['signal']=='PE') & (df_t['tk_bull']==True)) |
    ((df_t['signal']=='CE') & (df_t['tk_bull']==False))
]
tk_opposite = df_t[~df_t.index.isin(tk_aligned.index)]
if len(tk_aligned):
    print(f"    Aligned:  {len(tk_aligned)}t | WR {tk_aligned['win'].mean()*100:.1f}% | "
          f"Rs.{tk_aligned['pnl'].sum():,.0f} | Avg Rs.{tk_aligned['pnl'].mean():.0f}")
if len(tk_opposite):
    print(f"    Opposite: {len(tk_opposite)}t | WR {tk_opposite['win'].mean()*100:.1f}% | "
          f"Rs.{tk_opposite['pnl'].sum():,.0f} | Avg Rs.{tk_opposite['pnl'].mean():.0f}")

# vs CRT+MRC baseline
baseline_blank = 240468
print(f"\n  vs CRT+MRC baseline (ATM 30%): Rs.{baseline_blank:,.0f}")
print(f"  Ichimoku:                       Rs.{total_pnl:,.0f}")
diff = total_pnl - baseline_blank
print(f"  Difference:                    {'+'if diff>=0 else ''}Rs.{diff:,.0f}")

# ── Save ───────────────────────────────────────────────────────────────────────
df_t.to_csv(f'{OUT_DIR}/106_ichimoku_trades.csv', index=False)

# ── Equity chart ──────────────────────────────────────────────────────────────
df_t['date_dt'] = pd.to_datetime(df_t['date'].astype(str), format='%Y%m%d')
blank_daily = df_t.groupby('date_dt')['pnl'].sum().reset_index()
blank_daily.columns = ['date', 'blank_pnl']

base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base['date'] = pd.to_datetime(base['date'].astype(str), format='mixed')
base_daily = base.groupby('date')['pnl_conv'].sum().reset_index()
base_daily.columns = ['date', 'base_pnl']

all_dt = pd.DataFrame({'date': sorted(set(base_daily['date']) | set(blank_daily['date']))})
m = all_dt.merge(base_daily, on='date', how='left').merge(blank_daily, on='date', how='left')
m['base_pnl']  = m['base_pnl'].fillna(0)
m['blank_pnl'] = m['blank_pnl'].fillna(0)
m['comb_pnl']  = m['base_pnl'] + m['blank_pnl']
m['base_eq']   = m['base_pnl'].cumsum()
m['blank_eq']  = m['blank_pnl'].cumsum()
m['comb_eq']   = m['comb_pnl'].cumsum()

comb_total = round(m['comb_eq'].iloc[-1], 0)
comb_dd    = round((m['comb_eq'] - m['comb_eq'].cummax()).min(), 0)

print(f"\n  Combined (Base + Ichimoku): Rs.{comb_total:,.0f} | DD Rs.{comb_dd:,.0f}")

def eq_pts(series, dates):
    return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
            for d, v in zip(dates, series) if pd.notna(v)]

dates = m['date']
tv_json = {
    "isTvFormat": False, "candlestick": [], "volume": [],
    "lines": [
        {"id": "combined", "label": f"Combined Rs.{comb_total:,.0f}",
         "color": "#26a69a", "data": eq_pts(m['comb_eq'], dates), "seriesType": "line"},
        {"id": "base", "label": f"Base Rs.{int(m['base_pnl'].sum()):,.0f}",
         "color": "#0ea5e9", "data": eq_pts(m['base_eq'], dates), "seriesType": "line"},
        {"id": "ichimoku", "label": f"Ichimoku blank Rs.{total_pnl:,.0f}",
         "color": "#f59e0b", "data": eq_pts(m['blank_eq'], dates), "seriesType": "line"},
        {"id": "dd", "label": f"DD max Rs.{comb_dd:,.0f}",
         "color": "#ef5350",
         "data": eq_pts(m['comb_eq'] - m['comb_eq'].cummax(), dates),
         "seriesType": "baseline", "baseValue": 0, "isNewPane": True},
    ]
}

send_custom_chart("106_ichimoku_sell", tv_json,
                  title=f"Ichimoku Cloud Selling — Blank Days | {total}t | WR {wr}% | Combined Rs.{comb_total:,.0f}")

print(f"\n  Saved → {OUT_DIR}/106_ichimoku_trades.csv")
print("\nDone.")
