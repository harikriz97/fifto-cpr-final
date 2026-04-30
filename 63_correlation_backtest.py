"""
63_correlation_backtest.py
Deep correlation analysis — CPR + Camarilla + Pivots + VIX

For every v17a trade, compute 8 boolean confluence features:
  1. cpr_narrow          — CPR width < 0.15% (trending day expected)
  2. cpr_trend_aligned   — CPR pivot direction matches trade direction
  3. consec_aligned      — 3 consecutive days CPR in trade direction
  4. virgin_prev         — yesterday price stayed inside prev-prev CPR (virgin CPR)
  5. weekly_in_daily     — weekly CPR overlaps daily CPR (double-timeframe confluence)
  6. cam_aligned         — Cam L3/H3 inside CPR and direction matches trade
  7. cam_sr_overlap      — Cam H3 near S1 or L3 near R1 (double S/R cluster)
  8. vix_below_ma        — India VIX below 20-day MA (orderly market)

Then:
  - Show WR when each feature is True vs False
  - Conviction score = count of aligned features per trade
  - Lot sizing: score 0-1 = 1 lot, 2-3 = 2 lots, 4+ = 3 lots
  - Backtest conviction sizing vs flat 1-lot baseline
  - Charts: feature WR lift, score vs WR, equity comparison
"""
import sys, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
from plot_util import send_custom_chart

sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
from my_util import list_trading_dates, load_spot_data

DATA_ROOT = os.environ['INTER_SERVER_DATA_PATH']
SCALE     = 65 / 75   # rescale historical P&L (was computed at LOT=75)

def r2(v): return round(float(v), 2)

# ─────────────────────────────────────────────
# 0. Load base v17a trades
# ─────────────────────────────────────────────
df_all = pd.read_csv('data/56_combined_trades.csv', parse_dates=['date'])
v17a   = df_all[df_all['strategy'] == 'v17a'].copy().sort_values('date').reset_index(drop=True)
v17a['pnl65'] = v17a['pnl'] * SCALE
v17a['win']   = (v17a['pnl65'] > 0).astype(int)
BASE_WR  = round(v17a['win'].mean() * 100, 1)
BASE_PNL = r2(v17a['pnl65'].sum())
print(f"Base v17a: {len(v17a)}t | WR {BASE_WR}% | P&L Rs {BASE_PNL:,.0f}")

# ─────────────────────────────────────────────
# 1. Load daily OHLC for all dates (fast — just max/min/last/first)
# ─────────────────────────────────────────────
print("\nLoading daily OHLC (NIFTY)...")
dates    = list_trading_dates()
ohlc     = {}   # date_str -> (H, L, C, O)
for d in dates:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None or tks.empty:
        continue
    p = tks['price']
    ohlc[d] = (float(p.max()), float(p.min()), float(p.iloc[-1]), float(p.iloc[0]))
print(f"  Loaded OHLC for {len(ohlc)} days")

# ─────────────────────────────────────────────
# 2. Compute daily CPR / Cam / Pivot levels + features
# ─────────────────────────────────────────────
print("Computing features...")

# Precompute weekly OHLC (Mon–Fri buckets)
def week_key(d_str):
    dt = datetime.strptime(d_str, '%Y%m%d')
    mon = dt - timedelta(days=dt.weekday())
    return mon.strftime('%Y%m%d')

weekly_ohlc = {}   # week_start_str -> (H, L, C_last)
for d in sorted(ohlc.keys()):
    wk = week_key(d)
    h, l, c, o = ohlc[d]
    if wk not in weekly_ohlc:
        weekly_ohlc[wk] = [h, l, c]
    else:
        weekly_ohlc[wk][0] = max(weekly_ohlc[wk][0], h)
        weekly_ohlc[wk][1] = min(weekly_ohlc[wk][1], l)
        weekly_ohlc[wk][2] = c   # last day close of week

# Helper: get prev-week key
def prev_week_key(d_str):
    dt  = datetime.strptime(d_str, '%Y%m%d')
    mon = dt - timedelta(days=dt.weekday())
    return (mon - timedelta(days=7)).strftime('%Y%m%d')

features = {}
for i, d in enumerate(dates):
    if i < 5 or d not in ohlc:
        continue
    prev = dates[i-1]
    if prev not in ohlc:
        continue

    ph, pl, pc, _ = ohlc[prev]
    h, l, c, spot_o = ohlc[d]

    # ── Daily CPR (based on prev day) ──────────────
    pp   = (ph + pl + pc) / 3
    bc   = (ph + pl) / 2
    tc   = 2 * pp - bc
    width_pct = abs(tc - bc) / spot_o * 100

    # ── Standard pivots ────────────────────────────
    piv_r1 = 2 * pp - pl
    piv_r2 = pp + (ph - pl)
    piv_s1 = 2 * pp - ph
    piv_s2 = pp - (ph - pl)

    # ── Camarilla H3/L3/H4/L4 ─────────────────────
    rng    = ph - pl
    cam_h3 = pc + rng * 1.1 / 4
    cam_l3 = pc - rng * 1.1 / 4
    cam_h4 = pc + rng * 1.1 / 2
    cam_l4 = pc - rng * 1.1 / 2

    # ── Feature 1: CPR narrow ──────────────────────
    cpr_narrow = width_pct < 0.15

    # ── Feature 2: CPR trend direction ─────────────
    prev2 = dates[i-2] if i >= 2 else None
    cpr_trend = 'flat'
    if prev2 and prev2 in ohlc:
        p2h, p2l, p2c, _ = ohlc[prev2]
        prev_pp = (p2h + p2l + p2c) / 3
        if pp > prev_pp + 5:
            cpr_trend = 'bull'
        elif pp < prev_pp - 5:
            cpr_trend = 'bear'

    # ── Feature 3: Consecutive CPR direction ───────
    pivots_hist = []
    for j in range(max(1, i-3), i+1):
        dj = dates[j]
        dj_prev = dates[j-1]
        if dj in ohlc and dj_prev in ohlc:
            xh, xl, xc, _ = ohlc[dj_prev]
            pivots_hist.append((xh + xl + xc) / 3)
    consec_bull = (len(pivots_hist) >= 3 and
                   all(pivots_hist[k] < pivots_hist[k+1] for k in range(len(pivots_hist)-3, len(pivots_hist)-1)))
    consec_bear = (len(pivots_hist) >= 3 and
                   all(pivots_hist[k] > pivots_hist[k+1] for k in range(len(pivots_hist)-3, len(pivots_hist)-1)))

    # ── Feature 4: Virgin CPR (yesterday inside prev-prev CPR) ─
    virgin_prev = False
    if i >= 2 and prev2 and prev2 in ohlc:
        p2h, p2l, p2c, _ = ohlc[prev2]
        p2_pp = (p2h + p2l + p2c) / 3
        p2_bc = (p2h + p2l) / 2
        p2_tc = 2 * p2_pp - p2_bc
        # yesterday's high/low stayed within prev-prev CPR (±0.1% tolerance)
        tol = p2_pp * 0.001
        virgin_prev = (ph <= p2_tc + tol) and (pl >= p2_bc - tol)

    # ── Feature 5: Weekly CPR inside daily CPR ─────
    pw = prev_week_key(d)
    weekly_in_daily = False
    if pw in weekly_ohlc:
        wh, wl, wc = weekly_ohlc[pw]
        w_pp = (wh + wl + wc) / 3
        w_bc = (wh + wl) / 2
        w_tc = 2 * w_pp - w_bc
        # overlap: ranges share some zone (within 10pts)
        overlap_lo = max(w_bc, bc)
        overlap_hi = min(w_tc, tc)
        weekly_in_daily = (overlap_hi >= overlap_lo - 10)

    # ── Feature 6: Cam L3/H3 inside CPR (direction-aware stored separately) ─
    l3_in_cpr = (cam_l3 >= bc - 5) and (cam_l3 <= tc + 5)
    h3_in_cpr = (cam_h3 >= bc - 5) and (cam_h3 <= tc + 5)

    # ── Feature 7: Cam H3 near S1 or L3 near R1 (double S/R cluster) ─────
    h3_near_s1 = abs(cam_h3 - piv_s1) <= 25   # H3 ≈ S1 (double resistance from below)
    l3_near_r1 = abs(cam_l3 - piv_r1) <= 25   # L3 ≈ R1 (double support from above)

    features[d] = {
        'pp': r2(pp), 'tc': r2(tc), 'bc': r2(bc),
        'r1': r2(piv_r1), 'r2': r2(piv_r2), 's1': r2(piv_s1), 's2': r2(piv_s2),
        'cam_h3': r2(cam_h3), 'cam_l3': r2(cam_l3),
        'cpr_width_pct': r2(width_pct),
        'cpr_narrow':     cpr_narrow,
        'cpr_trend':      cpr_trend,
        'consec_bull':    consec_bull,
        'consec_bear':    consec_bear,
        'virgin_prev':    virgin_prev,
        'weekly_in_daily': weekly_in_daily,
        'l3_in_cpr':      l3_in_cpr,
        'h3_in_cpr':      h3_in_cpr,
        'h3_near_s1':     h3_near_s1,
        'l3_near_r1':     l3_near_r1,
        'spot_open':      r2(spot_o),
    }

feat_df = pd.DataFrame.from_dict(features, orient='index')
feat_df.index = pd.to_datetime(feat_df.index, format='%Y%m%d')
print(f"  Features computed: {len(feat_df)} days")

# ─────────────────────────────────────────────
# 3. Load VIX
# ─────────────────────────────────────────────
print("Loading VIX...")
vix_data = {}
for d in dates:
    path = os.path.join(DATA_ROOT, d, 'INDIAVIX.csv')
    if os.path.exists(path):
        try:
            tks = pd.read_csv(path, header=None, names=['date','time','price','vol','oi'])
            vix_data[d] = float(tks['price'].iloc[-1])
        except Exception:
            pass
vix_s  = pd.Series(vix_data)
vix_s.index = pd.to_datetime(vix_s.index, format='%Y%m%d')
vix_ma = vix_s.rolling(20, min_periods=5).mean()
feat_df['vix']          = vix_s
feat_df['vix_ma']       = vix_ma
feat_df['vix_below_ma'] = feat_df['vix'] < feat_df['vix_ma']
print(f"  VIX loaded for {vix_s.notna().sum()} days")

# ─────────────────────────────────────────────
# 4. Join features onto v17a trades
# ─────────────────────────────────────────────
v17a_f = v17a.copy()
v17a_f['date_idx'] = pd.to_datetime(v17a_f['date'])
feat_reset = feat_df.reset_index().rename(columns={'index': 'date_idx'})
v17a_f = v17a_f.merge(feat_reset, on='date_idx', how='left')
joined = v17a_f['cpr_width_pct'].notna().sum()
print(f"\nJoined {joined}/{len(v17a_f)} trades with features")

# ── Direction-aware composite features ────────
# PE trade = we expect index to NOT fall = bullish/neutral setup
# CE trade = we expect index to NOT rise = bearish/neutral setup
v17a_f['is_pe'] = v17a_f['opt'] == 'PE'
v17a_f['is_ce'] = v17a_f['opt'] == 'CE'

v17a_f['cpr_trend_aligned'] = (
    ((v17a_f['cpr_trend'] == 'bull') & v17a_f['is_pe']) |
    ((v17a_f['cpr_trend'] == 'bear') & v17a_f['is_ce'])
)
v17a_f['consec_aligned'] = (
    (v17a_f['consec_bull'] & v17a_f['is_pe']) |
    (v17a_f['consec_bear'] & v17a_f['is_ce'])
)
# cam_aligned: L3 in CPR = strong support = good for PE; H3 in CPR = strong resistance = good for CE
v17a_f['cam_aligned'] = (
    (v17a_f['l3_in_cpr'] & v17a_f['is_pe']) |
    (v17a_f['h3_in_cpr'] & v17a_f['is_ce'])
)
# cam_sr: L3 near R1 = double support above CPR (good for PE), H3 near S1 = double resistance below (good for CE)
v17a_f['cam_sr_overlap'] = (
    (v17a_f['l3_near_r1'] & v17a_f['is_pe']) |
    (v17a_f['h3_near_s1'] & v17a_f['is_ce'])
)

FEATURES = [
    'cpr_narrow',
    'cpr_trend_aligned',
    'consec_aligned',
    'virgin_prev',
    'weekly_in_daily',
    'cam_aligned',
    'cam_sr_overlap',
    'vix_below_ma',
]

# ─────────────────────────────────────────────
# 5. Feature-wise WR analysis
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FEATURE → WIN RATE LIFT  (base WR = {:.1f}%)".format(BASE_WR))
print("=" * 70)
print(f"  {'Feature':<22} {'N_true':>7} {'WR_true':>9} {'N_false':>7} {'WR_false':>9} {'Lift':>7}")
print(f"  {'-'*60}")

lift_map = {}
for feat in FEATURES:
    col = feat
    if col not in v17a_f.columns:
        continue
    t  = v17a_f[v17a_f[col] == True]
    f  = v17a_f[v17a_f[col] == False]
    if len(t) < 5:
        lift_map[feat] = 0
        continue
    wr_t = t['win'].mean() * 100
    wr_f = f['win'].mean() * 100 if len(f) > 0 else 0
    lift = wr_t - BASE_WR
    lift_map[feat] = r2(lift)
    flag = ' <-- BEST' if lift == max(lift_map.values()) else ''
    print(f"  {feat:<22} {len(t):>7} {wr_t:>8.1f}% {len(f):>7} {wr_f:>8.1f}% {lift:>+6.1f}%{flag}")

# ─────────────────────────────────────────────
# 6. Conviction score (only use features with positive lift)
# ─────────────────────────────────────────────
positive_feats = [f for f in FEATURES if lift_map.get(f, 0) > 0]
print(f"\n  Positive-lift features: {positive_feats}")

v17a_f['conv_score'] = v17a_f[positive_feats].fillna(False).sum(axis=1).astype(int)

print("\n" + "=" * 70)
print("  CONVICTION SCORE → WIN RATE")
print("=" * 70)
print(f"  {'Score':<8} {'Trades':>7} {'WR':>9} {'Avg P&L':>10} {'Total P&L':>12}")
print(f"  {'-'*50}")
for score in sorted(v17a_f['conv_score'].unique()):
    sub = v17a_f[v17a_f['conv_score'] == score]
    wr  = sub['win'].mean() * 100
    avg = sub['pnl65'].mean()
    tot = sub['pnl65'].sum()
    bar = '#' * int(wr / 5)
    print(f"  {score:<8} {len(sub):>7} {wr:>8.1f}% {avg:>10,.0f} {tot:>12,.0f}  {bar}")

# ─────────────────────────────────────────────
# 7. Conviction-based lot sizing backtest
# 0-1 signals → 1 lot  (conservative, uncertain)
# 2-3 signals → 2 lots (good conviction)
# 4+  signals → 3 lots (strong confluence)
# ─────────────────────────────────────────────
def lots(score):
    if score >= 4: return 3
    if score >= 2: return 2
    return 1

v17a_f['conv_lots'] = v17a_f['conv_score'].apply(lots)
v17a_f['conv_pnl']  = v17a_f['pnl65'] * v17a_f['conv_lots']

v17a_s    = v17a_f.sort_values('date').reset_index(drop=True)
eq_base   = v17a_s['pnl65'].cumsum()
eq_conv   = v17a_s['conv_pnl'].cumsum()
dd_base   = r2((eq_base - eq_base.cummax()).min())
dd_conv   = r2((eq_conv - eq_conv.cummax()).min())
pnl_base  = r2(eq_base.iloc[-1])
pnl_conv  = r2(eq_conv.iloc[-1])
improvement = r2(pnl_conv - pnl_base)

lot_dist = dict(v17a_f['conv_lots'].value_counts().sort_index())

print("\n" + "=" * 70)
print("  CONVICTION LOT SIZING vs FLAT BASELINE")
print("=" * 70)
print(f"  Flat 1-lot:      {len(v17a_s):>4}t | WR {BASE_WR:>5.1f}% | P&L Rs {pnl_base:>10,.0f} | DD Rs {dd_base:>10,.0f}")

conv_wr = r2((v17a_f['conv_pnl'] > 0).mean() * 100)
print(f"  Conviction:      {len(v17a_s):>4}t | WR {conv_wr:>5.1f}% | P&L Rs {pnl_conv:>10,.0f} | DD Rs {dd_conv:>10,.0f}")
print(f"  Improvement:                              Rs {improvement:>+10,.0f}")
print(f"  Lot split: 1-lot={lot_dist.get(1,0)}t  2-lot={lot_dist.get(2,0)}t  3-lot={lot_dist.get(3,0)}t")

# Year-wise conviction vs baseline
print("\n  Year-wise breakdown:")
v17a_f['year'] = pd.to_datetime(v17a_f['date']).dt.year
print(f"  {'Year':<6} {'t':>5} {'Base P&L':>12} {'Conv P&L':>12} {'Lift':>10}")
for yr, g in v17a_f.groupby('year'):
    bp = r2(g['pnl65'].sum())
    cp = r2(g['conv_pnl'].sum())
    print(f"  {yr:<6} {len(g):>5} {bp:>12,.0f} {cp:>12,.0f} {cp-bp:>+10,.0f}")

# ─────────────────────────────────────────────
# 8. Charts
# ─────────────────────────────────────────────
ts_base = int(pd.Timestamp('2021-01-01').timestamp())

# Chart 1: Feature WR lift bar
feat_bar = []
for i, feat in enumerate(FEATURES):
    lift = lift_map.get(feat, 0)
    wr_t = BASE_WR + lift
    feat_bar.append({
        'time':  ts_base + i * 86400 * 30,
        'value': round(wr_t, 1),
        'color': '#26a69a' if lift >= 0 else '#ef5350',
        'label': feat.replace('_',' ')
    })
tv1 = {
    'lines': [{'id':'feat_wr','label':'WR when feature=True','seriesType':'bar',
               'data': feat_bar,
               'xLabels': [b['label'] for b in feat_bar]}],
    'candlestick': [], 'volume': [], 'isTvFormat': False
}
send_custom_chart('feature_wr', tv1,
    title=f'Feature Win Rate | Base WR = {BASE_WR}% (dashed line)')
print("\n📊 Feature WR chart sent")

# Chart 2: Score vs WR scatter (as bar)
score_bar = []
ts2 = int(pd.Timestamp('2021-06-01').timestamp())
for score in sorted(v17a_f['conv_score'].unique()):
    sub = v17a_f[v17a_f['conv_score'] == score]
    wr  = round(sub['win'].mean() * 100, 1)
    score_bar.append({
        'time':  ts2 + int(score) * 86400 * 60,
        'value': wr,
        'color': '#26a69a' if wr >= BASE_WR else '#f59e0b',
        'label': f'Score {score} ({len(sub)}t)'
    })
tv2 = {
    'lines': [{'id':'score_wr','label':'WR by conviction score','seriesType':'bar',
               'data': score_bar,
               'xLabels': [b['label'] for b in score_bar]}],
    'candlestick': [], 'volume': [], 'isTvFormat': False
}
send_custom_chart('score_wr', tv2,
    title=f'Conviction Score vs Win Rate | Base={BASE_WR}%')
print("📊 Conviction score WR chart sent")

# Chart 3: Equity — baseline vs conviction sizing
def to_line(label, color, s):
    return {
        'id':    label.replace(' ','_'),
        'label': label, 'color': color,
        'data':  [{'time': int(pd.Timestamp(row.date).timestamp()), 'value': round(float(v), 2)}
                  for (_, row), v in zip(v17a_s.iterrows(), s.values)]
    }

tv3 = {
    'lines': [
        to_line('Flat 1-lot',          '#8b949e', eq_base),
        to_line('Conviction (1/2/3x)', '#26a69a', eq_conv),
    ],
    'candlestick': [], 'volume': [], 'isTvFormat': False
}
send_custom_chart('conviction_equity', tv3,
    title=f'Conviction Sizing vs Flat | +Rs {improvement:+,.0f} improvement')
print("📊 Conviction equity chart sent")

# Chart 4: Year-wise lift bars
yr_data = []
ts3 = int(pd.Timestamp('2021-01-01').timestamp())
for i, (yr, g) in enumerate(v17a_f.groupby('year')):
    bp = r2(g['pnl65'].sum())
    cp = r2(g['conv_pnl'].sum())
    yr_data.append({'year': yr, 'base': bp, 'conv': cp, 'lift': cp - bp})

base_bars = [{'time': ts3 + i*86400*365, 'value': row['base'],
              'color': '#8b949e', 'label': str(row['year'])}
             for i, row in enumerate(yr_data)]
conv_bars = [{'time': ts3 + i*86400*365 + 86400*30, 'value': row['conv'],
              'color': '#26a69a', 'label': str(row['year'])}
             for i, row in enumerate(yr_data)]

tv4 = {
    'lines': [
        {'id':'base_yr','label':'Flat 1-lot','seriesType':'bar','data': base_bars,
         'xLabels': [b['label'] for b in base_bars]},
        {'id':'conv_yr','label':'Conviction','seriesType':'bar','data': conv_bars,
         'xLabels': [b['label'] for b in conv_bars]},
    ],
    'candlestick': [], 'volume': [], 'isTvFormat': False
}
send_custom_chart('year_lift', tv4,
    title='Year-wise: Flat vs Conviction Sizing')
print("📊 Year-wise lift chart sent")

print("\nAll done!")
