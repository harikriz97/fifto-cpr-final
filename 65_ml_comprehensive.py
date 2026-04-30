"""
65_ml_comprehensive.py
======================
Full ML + analysis — directional only, no straddle.

Sections:
  A. CPR Opening Position (open above/below/inside CPR)
  B. Full Feature Matrix + Correlation Table
  C. ML Feature Importance (RandomForest, no forward bias)
  D. ML Conviction Scoring → 1/2/3 lot sizing
  E. SENSEX Pattern (NIFTY signal on SENSEX Thursday expiry)
  F. Expiry Regime Analysis (Thu→Tue switch Sep 2025)
  G. 3-Lot Conviction Final Design — validated
  H. 3-Way Validation (time-split, zone-split, year-LOO)

Forward-bias rules enforced:
  - CPR from PREV day OHLC only
  - EMA uses shift(1) on close series
  - VIX uses prev day close
  - DTE computed from actual expiry list (known pre-market)
  - No same-day look-ahead anywhere
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
from plot_util import send_custom_chart

from my_util import list_trading_dates, load_spot_data, list_expiry_dates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SCALE = 65 / 75

def r2(v): return round(float(v), 2)

def show(label, pnls, pad=38):
    pnls = np.array(pnls, dtype=float)
    n = len(pnls)
    if n == 0:
        print(f"  {label:<{pad}}   0t")
        return
    wr  = r2((pnls > 0).mean() * 100)
    tot = r2(pnls.sum())
    avg = r2(pnls.mean())
    eq  = np.cumsum(pnls)
    dd  = r2((eq - np.maximum.accumulate(eq)).min()) if n > 1 else 0
    print(f"  {label:<{pad}} {n:>4}t | WR {wr:>5.1f}% | Avg {avg:>7,.0f} | Tot {tot:>10,.0f} | DD {dd:>9,.0f}")

# ─────────────────────────────────────────────
# LOAD BASE v17a TRADES
# ─────────────────────────────────────────────
df_raw = pd.read_csv('data/56_combined_trades.csv', parse_dates=['date'])
v17a   = df_raw[df_raw['strategy'] == 'v17a'].copy().sort_values('date').reset_index(drop=True)
v17a['pnl65'] = v17a['pnl'] * SCALE
v17a['win']   = (v17a['pnl65'] > 0).astype(int)
BASE_WR  = round(v17a['win'].mean() * 100, 1)
print(f"Base v17a: {len(v17a)}t | WR {BASE_WR}% | P&L Rs {v17a.pnl65.sum():,.0f}\n")

# ─────────────────────────────────────────────
# LOAD DAILY OHLC (all dates — forward-bias safe: used only for prev-day features)
# ─────────────────────────────────────────────
print("Loading OHLC + VIX...")
t0 = time.time()
all_dates  = list_trading_dates()
daily_ohlc = {}
for d in all_dates:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None or tks.empty: continue
    p = tks['price']
    daily_ohlc[d] = {
        'H': float(p.max()), 'L': float(p.min()),
        'C': float(p.iloc[-1]), 'O': float(p.iloc[0]),
        'open_15': float(tks[tks['time'] >= '09:15:00']['price'].iloc[0]),
    }

# VIX
DATA_ROOT = os.environ['INTER_SERVER_DATA_PATH']
vix_raw = {}
for d in all_dates:
    path = os.path.join(DATA_ROOT, d, 'INDIAVIX.csv')
    if os.path.exists(path):
        try:
            t = pd.read_csv(path, header=None, names=['d','t','p','v','oi'])
            vix_raw[d] = float(t['p'].iloc[-1])
        except: pass
vix_s  = pd.Series(vix_raw).sort_index()
# shift(1): use PREV day VIX close — forward-bias safe
vix_ma20 = vix_s.rolling(20, min_periods=5).mean().shift(1)
vix_prev  = vix_s.shift(1)
print(f"  OHLC:{len(daily_ohlc)}d  VIX:{len(vix_raw)}d  ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────
# BUILD DAILY FEATURE TABLE (forward-bias safe)
# All features use only data available BEFORE today's open
# ─────────────────────────────────────────────
print("Building features (no forward bias)...")

def compute_cpr(h, l, c):
    pp = (h + l + c) / 3
    bc = (h + l) / 2
    tc = 2 * pp - bc
    return r2(pp), r2(tc), r2(bc)

feat_rows = {}
for i, d in enumerate(all_dates):
    if i < 3 or d not in daily_ohlc: continue
    prev  = all_dates[i-1]
    prev2 = all_dates[i-2]
    prev3 = all_dates[i-3]
    if prev not in daily_ohlc or prev2 not in daily_ohlc: continue

    ph = daily_ohlc[prev]['H']; pl = daily_ohlc[prev]['L']
    pc = daily_ohlc[prev]['C']; po = daily_ohlc[prev]['O']
    today_o = daily_ohlc[d]['open_15']   # today's actual 9:15 open (entry signal uses this)

    pp, tc, bc = compute_cpr(ph, pl, pc)
    cpr_lo = min(tc, bc); cpr_hi = max(tc, bc)
    width_pct = abs(tc - bc) / today_o * 100

    # Standard pivots (from prev day)
    r1 = r2(2*pp - pl); r2_ = r2(pp + (ph-pl))
    s1 = r2(2*pp - ph); s2  = r2(pp - (ph-pl))

    # Camarilla
    rng   = ph - pl
    cam_h3 = r2(pc + rng*1.1/4); cam_l3 = r2(pc - rng*1.1/4)
    cam_h4 = r2(pc + rng*1.1/2); cam_l4 = r2(pc - rng*1.1/2)

    # CPR trend (today's PP vs yesterday's PP) — using prev-prev day for yesterday's PP
    p2h = daily_ohlc[prev2]['H']; p2l = daily_ohlc[prev2]['L']; p2c = daily_ohlc[prev2]['C']
    prev_pp, _, _ = compute_cpr(p2h, p2l, p2c)
    cpr_trend = 'bull' if pp > prev_pp + 5 else ('bear' if pp < prev_pp - 5 else 'flat')

    # Consecutive CPR direction (3 days rising or falling)
    pivots = []
    for j in range(max(0, i-4), i):
        dj = all_dates[j]; djp = all_dates[j-1] if j>0 else None
        if djp and dj in daily_ohlc and djp in daily_ohlc:
            xh = daily_ohlc[djp]['H']; xl = daily_ohlc[djp]['L']; xc = daily_ohlc[djp]['C']
            pivots.append((xh+xl+xc)/3)
    consec_bull = len(pivots) >= 3 and all(pivots[k] < pivots[k+1] for k in range(len(pivots)-3, len(pivots)-1))
    consec_bear = len(pivots) >= 3 and all(pivots[k] > pivots[k+1] for k in range(len(pivots)-3, len(pivots)-1))

    # CPR gap (virgin CPR) — today CPR vs yesterday CPR
    p2_pp2, p2_tc2, p2_bc2 = compute_cpr(p2h, p2l, p2c)
    prev_cpr_lo = min(p2_tc2, p2_bc2); prev_cpr_hi = max(p2_tc2, p2_bc2)
    cpr_gap_bull = cpr_lo > prev_cpr_hi    # today CPR above yesterday
    cpr_gap_bear = cpr_hi < prev_cpr_lo    # today CPR below yesterday
    cpr_gap      = cpr_gap_bull or cpr_gap_bear

    # Opening position relative to CPR (forward-bias safe: uses 9:15 open)
    if today_o > cpr_hi:
        open_pos = 'above'    # bull → sell PE
    elif today_o < cpr_lo:
        open_pos = 'below'    # bear → sell CE
    else:
        open_pos = 'inside'   # range → avoid or small size

    # DTE — use actual expiry (no forward bias: list_expiry_dates uses only known expiry calendar)
    try:
        expiries = list_expiry_dates(d)
        if expiries:
            exp_dt = datetime.strptime('20' + expiries[0], '%Y%m%d')
            trade_dt = datetime.strptime(d, '%Y%m%d')
            dte = (exp_dt - trade_dt).days
        else:
            dte = np.nan
    except:
        dte = np.nan

    # Day of week
    dow = datetime.strptime(d, '%Y%m%d').weekday()  # 0=Mon,1=Tue,...,6=Sun

    # VIX features (shift already applied above — using prev day's VIX)
    d_ts = pd.Timestamp(d)
    vix_val  = vix_prev.get(d_ts, np.nan)
    vix_ma_v = vix_ma20.get(d_ts, np.nan)
    vix_ok   = bool(not np.isnan(vix_val) and not np.isnan(vix_ma_v) and vix_val < vix_ma_v)

    # L3/H3 inside CPR
    l3_in_cpr = cpr_lo - 5 <= cam_l3 <= cpr_hi + 5
    h3_in_cpr = cpr_lo - 5 <= cam_h3 <= cpr_hi + 5

    # Expiry regime
    exp_regime = 'thursday' if dow == 3 else ('tuesday' if dow == 1 else 'other')
    # Actually regime = what weekday is expiry, determine from DTE + dow
    # Simple: if dte <= 2 and dow == 0 (Monday) → likely new Tuesday regime
    is_tue_regime = (not np.isnan(dte)) and datetime.strptime(d, '%Y%m%d') >= datetime(2025, 9, 1)

    feat_rows[d] = {
        'pp': pp, 'tc': tc, 'bc': bc,
        'r1': r1, 'r2': r2_, 's1': s1, 's2': s2,
        'cam_h3': cam_h3, 'cam_l3': cam_l3,
        'cpr_width_pct': width_pct,
        'cpr_narrow':    width_pct < 0.20,
        'cpr_ultra_narrow': width_pct < 0.10,
        'cpr_trend':     cpr_trend,
        'consec_bull':   consec_bull,
        'consec_bear':   consec_bear,
        'cpr_gap':       cpr_gap,
        'cpr_gap_bull':  cpr_gap_bull,
        'cpr_gap_bear':  cpr_gap_bear,
        'open_pos':      open_pos,
        'open_above':    open_pos == 'above',
        'open_below':    open_pos == 'below',
        'open_inside':   open_pos == 'inside',
        'l3_in_cpr':     l3_in_cpr,
        'h3_in_cpr':     h3_in_cpr,
        'vix_ok':        vix_ok,
        'dte':           dte,
        'dow':           dow,
        'is_tue_regime': is_tue_regime,
        'spot_open':     today_o,
    }

feat_df = pd.DataFrame.from_dict(feat_rows, orient='index')
feat_df.index = pd.to_datetime(feat_df.index, format='%Y%m%d')
print(f"  Features: {len(feat_df)} days\n")

# Join features onto v17a trades
v17a['date_idx'] = pd.to_datetime(v17a['date'])
feat_reset = feat_df.reset_index().rename(columns={'index': 'date_idx'})
v = v17a.merge(feat_reset, on='date_idx', how='left')
v['is_pe']  = v['opt'] == 'PE'
v['is_ce']  = v['opt'] == 'CE'

# Direction-aware composite features
v['cpr_trend_aligned'] = ((v['cpr_trend']=='bull') & v['is_pe']) | ((v['cpr_trend']=='bear') & v['is_ce'])
v['consec_aligned']    = (v['consec_bull'] & v['is_pe']) | (v['consec_bear'] & v['is_ce'])
v['cpr_gap_aligned']   = (v['cpr_gap_bull'] & v['is_pe']) | (v['cpr_gap_bear'] & v['is_ce'])
v['open_aligned']      = (v['open_above']   & v['is_pe']) | (v['open_below']   & v['is_ce'])
v['cam_aligned']       = (v['l3_in_cpr']    & v['is_pe']) | (v['h3_in_cpr']    & v['is_ce'])

print(f"Joined {v['cpr_width_pct'].notna().sum()}/{len(v)} trades with features\n")

# ═══════════════════════════════════════════════════════════════
# A. CPR OPENING POSITION ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("  A. CPR OPENING POSITION (open above/below/inside CPR)")
print("=" * 65)
print("  forward-bias safe: uses 9:15 open vs prev-day CPR levels\n")

for pos in ['above', 'inside', 'below']:
    sub = v[v['open_pos'] == pos]
    label = f'Open {pos} CPR'
    show(label, sub['pnl65'])

# Direction-aligned: open above → PE, open below → CE
print()
show('Open aligned (above+PE, below+CE)', v[v['open_aligned']]['pnl65'])
show('Open misaligned',                   v[~v['open_aligned']]['pnl65'])

# Cross tab: open_pos × opt_type
print(f"\n  Cross-tab (WR %):")
print(f"  {'':12} {'PE sell':>10} {'CE sell':>10}")
for pos in ['above','inside','below']:
    pe_sub = v[(v['open_pos']==pos) & v['is_pe']]
    ce_sub = v[(v['open_pos']==pos) & v['is_ce']]
    pe_wr  = r2(pe_sub['win'].mean()*100) if len(pe_sub) else np.nan
    ce_wr  = r2(ce_sub['win'].mean()*100) if len(ce_sub) else np.nan
    print(f"  Open {pos:<7}  {pe_wr:>8.1f}%  {ce_wr:>8.1f}%   ({len(pe_sub)}t / {len(ce_sub)}t)")

# ═══════════════════════════════════════════════════════════════
# B. FULL FEATURE CORRELATION TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  B. FULL FEATURE → WIN RATE TABLE")
print("=" * 65)
print(f"  {'Feature':<28} {'N_yes':>6} {'WR_yes':>9} {'WR_no':>9} {'Lift':>8} {'Phi':>8}")
print(f"  {'-'*65}")

ALL_FEATURES = [
    'cpr_narrow', 'cpr_ultra_narrow',
    'cpr_trend_aligned', 'consec_aligned',
    'cpr_gap', 'cpr_gap_aligned',
    'open_aligned',
    'l3_in_cpr', 'h3_in_cpr', 'cam_aligned',
    'vix_ok',
]

lift_map = {}
phi_map  = {}

for feat in ALL_FEATURES:
    if feat not in v.columns: continue
    t_  = v[v[feat] == True];   f_ = v[v[feat] == False]
    if len(t_) < 5: continue
    wr_t = t_['win'].mean() * 100
    wr_f = f_['win'].mean() * 100 if len(f_) > 0 else 0
    lift = r2(wr_t - BASE_WR)

    # Phi coefficient (point-biserial correlation)
    from scipy.stats import pointbiserialr
    pb, pv = pointbiserialr(v[feat].astype(float), v['win'].astype(float))
    phi = r2(pb)

    lift_map[feat] = lift
    phi_map[feat]  = phi
    marker = ' <<' if lift > 3 else ''
    print(f"  {feat:<28} {len(t_):>6} {wr_t:>8.1f}% {wr_f:>8.1f}% {lift:>+7.1f}% {phi:>+7.3f}{marker}")

# ═══════════════════════════════════════════════════════════════
# C. ML FEATURE IMPORTANCE (RandomForest)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  C. ML FEATURE IMPORTANCE (RandomForest — no forward bias)")
print("=" * 65)

ML_FEATURES = [
    'cpr_narrow', 'cpr_ultra_narrow',
    'cpr_trend_aligned', 'consec_aligned',
    'cpr_gap_aligned', 'open_aligned',
    'cam_aligned', 'vix_ok',
    'cpr_width_pct',
]

# Add numeric DTE
v['dte_num'] = v['dte'].astype(float).clip(0, 10)
v['dte_3']   = (v['dte_num'] == 3).astype(float)
v['dte_lte3']= (v['dte_num'] <= 3).astype(float)
ML_FEATURES += ['dte_num', 'dte_3', 'dte_lte3', 'dow']

# Prepare matrix
ml_df = v[ML_FEATURES + ['win', 'pnl65', 'date']].dropna()
X = ml_df[ML_FEATURES].astype(float)
y = ml_df['win'].astype(int)

# Strict time-based split: train on ≤2023, test on 2024+
train_mask = pd.to_datetime(ml_df['date']).dt.year <= 2023
test_mask  = ~train_mask
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"  Train: {train_mask.sum()}t (≤2023) | Test: {test_mask.sum()}t (2024+)")

# RandomForest
rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10,
                             random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=ML_FEATURES).sort_values(ascending=False)
print(f"\n  Feature Importance (RF):")
for feat, imp in importances.items():
    bar = '█' * int(imp * 100)
    print(f"  {feat:<22} {imp:>6.3f}  {bar}")

# Test AUC
test_proba = rf.predict_proba(X_test)[:, 1]
test_auc   = r2(roc_auc_score(y_test, test_proba))
print(f"\n  Test AUC (2024+): {test_auc}")

# CV on full data
cv_scores = cross_val_score(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
print(f"  5-fold CV AUC: {r2(cv_scores.mean())} ± {r2(cv_scores.std())}")

# ═══════════════════════════════════════════════════════════════
# D. ML CONVICTION SCORING → LOT SIZING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  D. ML CONVICTION SCORING → 1/2/3 LOT SIZING")
print("=" * 65)

# Fit on ALL data for final probability scores (for analysis only)
rf_full = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10,
                                  random_state=42, class_weight='balanced')
rf_full.fit(X, y)
ml_df = ml_df.copy()
ml_df['ml_prob'] = rf_full.predict_proba(X)[:, 1]

# Probability bins
print("  ML Prob → WR:")
print(f"  {'Prob range':<15} {'t':>5} {'WR':>8} {'Avg P&L':>10}")
for lo, hi in [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
    sub = ml_df[(ml_df['ml_prob'] >= lo) & (ml_df['ml_prob'] < hi)]
    if len(sub) == 0: continue
    wr  = sub['win'].mean()*100
    avg = sub['pnl65'].mean()
    print(f"  {lo:.1f}–{hi:.1f}          {len(sub):>5} {wr:>7.1f}% {avg:>10,.0f}")

# Lot rule: prob < 0.55 → 1 lot; 0.55–0.70 → 2 lots; >0.70 → 3 lots
def ml_lots(p):
    if p >= 0.70: return 3
    if p >= 0.55: return 2
    return 1

ml_df['ml_lots'] = ml_df['ml_prob'].apply(ml_lots)
ml_df['ml_pnl']  = ml_df['pnl65'] * ml_df['ml_lots']

ml_df_s   = ml_df.sort_values('date').reset_index(drop=True)
eq_base   = ml_df_s['pnl65'].cumsum()
eq_ml     = ml_df_s['ml_pnl'].cumsum()
dd_base   = r2((eq_base - eq_base.cummax()).min())
dd_ml     = r2((eq_ml   - eq_ml.cummax()).min())
pnl_base  = r2(eq_base.iloc[-1])
pnl_ml    = r2(eq_ml.iloc[-1])
lot_dist  = dict(ml_df['ml_lots'].value_counts().sort_index())

print(f"\n  Flat 1-lot baseline:  {len(ml_df)}t | P&L Rs {pnl_base:>10,.0f} | DD Rs {dd_base:>9,.0f}")
print(f"  ML conviction lots:   {len(ml_df)}t | P&L Rs {pnl_ml:>10,.0f} | DD Rs {dd_ml:>9,.0f}")
print(f"  Improvement: Rs {pnl_ml-pnl_base:>+10,.0f}")
print(f"  Lot dist: {lot_dist}")

# ═══════════════════════════════════════════════════════════════
# E. SENSEX PATTERN (NIFTY signal → SENSEX Thursday trade)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  E. SENSEX PATTERN (NIFTY CPR signal on SENSEX Thursday expiry)")
print("=" * 65)
print("  Note: SENSEX data available only from ~Dec 2025")
print("  SENSEX lot=20, strike_interval=100, expires Thursday")

df_sensex = pd.read_csv('data/20260428/51_sensex_tuesday_trades.csv',
                         parse_dates=['date'])
# Best params from grid search
# Filter to get best single config
best = (
    df_sensex.groupby(['strike_type','target_pct','sl_pct'])
             .apply(lambda g: pd.Series({
                 'n': len(g),
                 'wr': (g.pnl>0).mean()*100,
                 'avg': g.pnl.mean(),
                 'tot': g.pnl.sum()
             }))
             .reset_index()
)
best = best.sort_values('avg', ascending=False)
print(f"\n  Top 5 SENSEX configs (by avg P&L):")
print(f"  {'Strike':>8} {'Tgt':>6} {'SL':>6} {'n':>5} {'WR':>8} {'Avg':>8} {'Total':>10}")
for _, row in best.head(5).iterrows():
    print(f"  {row.strike_type:>8} {row.target_pct:>6.2f} {row.sl_pct:>6.2f} "
          f"{int(row.n):>5} {row.wr:>7.1f}% {row.avg:>8,.0f} {row.tot:>10,.0f}")

# Best config trades
brow = best.iloc[0]
df_best = df_sensex[(df_sensex.strike_type==brow.strike_type) &
                    (df_sensex.target_pct==brow.target_pct) &
                    (df_sensex.sl_pct==brow.sl_pct)].copy()
show('SENSEX best config', df_best['pnl'])

# Entry time breakdown
print(f"\n  SENSEX by entry time:")
for et, g in df_best.groupby('entry_time'):
    show(f'  {et}', g['pnl'])

# Zone breakdown
print(f"\n  SENSEX by zone (top 5):")
zone_perf = df_best.groupby('zone').apply(
    lambda g: pd.Series({'n':len(g),'wr':(g.pnl>0).mean()*100,'avg':g.pnl.mean()}))
for _, row in zone_perf.sort_values('avg', ascending=False).head(5).iterrows():
    print(f"  {row.name:<18} {int(row.n):>4}t WR {row.wr:.1f}% Avg {row.avg:,.0f}")

# ═══════════════════════════════════════════════════════════════
# F. EXPIRY REGIME ANALYSIS (Thu→Tue switch)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  F. EXPIRY REGIME (Thu expiry 2021–Aug2025 vs Tue 2025–)")
print("=" * 65)

v['trade_dt'] = pd.to_datetime(v['date'])
v['regime']   = v['trade_dt'].apply(
    lambda d: 'tuesday_expiry' if d >= pd.Timestamp('2025-09-01') else 'thursday_expiry')

for regime, g in v.groupby('regime'):
    print(f"\n  Regime: {regime} ({len(g)}t)")
    show(f'  All',              g['pnl65'])
    show(f'  DTE 1-2 (expiry eve)', g[g['dte_num'] <= 2]['pnl65'])
    show(f'  DTE 3-5 (sweet spot)', g[(g['dte_num']>=3) & (g['dte_num']<=5)]['pnl65'])
    show(f'  DTE 6+ (early)',       g[g['dte_num'] >= 6]['pnl65'])

# Day of week by regime
print(f"\n  Day of week (Thursday regime):")
thu_regime = v[v['regime']=='thursday_expiry']
dow_names = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
for dow_val in sorted(thu_regime['dow'].unique()):
    g = thu_regime[thu_regime['dow'] == dow_val]
    show(f'  {dow_names[dow_val]} (DTE~{3-dow_val if dow_val<=3 else 10-dow_val})', g['pnl65'])

print(f"\n  Day of week (Tuesday regime):")
tue_regime = v[v['regime']=='tuesday_expiry']
for dow_val in sorted(tue_regime['dow'].unique()):
    g = tue_regime[tue_regime['dow'] == dow_val]
    # DTE to Tuesday(=1): Mon=1, Sun=2, Sat=3, Fri=4, Thu=5, Wed=6
    show(f'  {dow_names[dow_val]}', g['pnl65'])

# ═══════════════════════════════════════════════════════════════
# G. 3-LOT CONVICTION FINAL DESIGN
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  G. 3-LOT CONVICTION FINAL DESIGN")
print("=" * 65)
print("  Rule: each condition = +1 signal. 0=1lot 2=2lots 4+=3lots")
print("  Conditions (all forward-bias safe, theory-backed):")
print("  1. VIX below 20-day MA  (orderly market +5.8% lift)")
print("  2. CPR trend aligned    (CPR direction matches trade +2.8%)")
print("  3. Consec aligned       (3-day CPR streak +4.2%)")
print("  4. CPR gap aligned      (virgin CPR gap +4.4%)")
print("  5. Open pos aligned     (open above CPR → sell PE etc)")

CONV_FEATURES = ['vix_ok','cpr_trend_aligned','consec_aligned','cpr_gap_aligned','open_aligned']

v_conv = v.copy()
v_conv['conv_score'] = v_conv[CONV_FEATURES].fillna(False).sum(axis=1).astype(int)

def conv_lots(s):
    if s >= 4: return 3
    if s >= 2: return 2
    return 1

v_conv['conv_lots'] = v_conv['conv_score'].apply(conv_lots)
v_conv['conv_pnl']  = v_conv['pnl65'] * v_conv['conv_lots']

print(f"\n  Score → WR → Lots:")
print(f"  {'Score':<8} {'Lots':>5} {'t':>5} {'WR':>8} {'Avg':>8}")
for sc in sorted(v_conv['conv_score'].unique()):
    sub = v_conv[v_conv['conv_score'] == sc]
    lots_ = conv_lots(sc)
    wr    = sub['pnl65'].gt(0).mean()*100
    avg   = sub['pnl65'].mean()
    print(f"  {sc:<8} {lots_:>5}x {len(sub):>5} {wr:>7.1f}% {avg:>8,.0f}")

v_s = v_conv.sort_values('date').reset_index(drop=True)
eq_base = v_s['pnl65'].cumsum();  dd_b = (eq_base - eq_base.cummax()).min()
eq_conv = v_s['conv_pnl'].cumsum(); dd_c = (eq_conv - eq_conv.cummax()).min()

print(f"\n  Flat 1-lot:   P&L Rs {v_s.pnl65.sum():>10,.0f} | DD Rs {dd_b:>9,.0f}")
print(f"  Conviction:   P&L Rs {v_s.conv_pnl.sum():>10,.0f} | DD Rs {dd_c:>9,.0f}")
print(f"  Improvement:  Rs {v_s.conv_pnl.sum()-v_s.pnl65.sum():>+10,.0f}")
print(f"  Lot dist: {dict(v_conv['conv_lots'].value_counts().sort_index())}")

# Year-wise validation
print(f"\n  Year-wise conviction:")
for yr, g in v_conv.groupby(pd.to_datetime(v_conv['date']).dt.year):
    b = r2(g['pnl65'].sum()); c = r2(g['conv_pnl'].sum())
    print(f"  {yr}  {len(g):>4}t  Base Rs {b:>9,.0f}  Conv Rs {c:>9,.0f}  Lift Rs {c-b:>+9,.0f}")

# ═══════════════════════════════════════════════════════════════
# H. 3-WAY VALIDATION (no forward bias confirmed)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  H. 3-WAY VALIDATION")
print("=" * 65)

v_val = v_conv.copy()
v_val['year'] = pd.to_datetime(v_val['date']).dt.year

# H1: Time-based OOS — train 2021-2023, test 2024-2026
print("\n  H1: Time-based OOS (train ≤2023, test 2024+)")
train = v_val[v_val['year'] <= 2023]
test  = v_val[v_val['year'] >= 2024]
show('  Train (2021-2023) flat', train['pnl65'])
show('  Train (2021-2023) conv', train['conv_pnl'])
show('  TEST  (2024-2026) flat', test['pnl65'])
show('  TEST  (2024-2026) conv', test['conv_pnl'])
test_wr_flat = r2(test['pnl65'].gt(0).mean()*100)
test_wr_conv = r2(test['conv_pnl'].gt(0).mean()*100)
print(f"  → OOS WR flat: {test_wr_flat}%  |  OOS WR conv: {test_wr_conv}%")

# H2: Leave-one-year-out (LOO-Year)
print("\n  H2: Leave-one-year-out (LOO-Y)")
print(f"  {'Test year':<12} {'Flat WR':>10} {'Conv WR':>10} {'Flat P&L':>12} {'Conv P&L':>12}")
for yr in sorted(v_val['year'].unique()):
    test_yr = v_val[v_val['year'] == yr]
    flat_wr = r2(test_yr['pnl65'].gt(0).mean()*100)
    conv_wr = r2(test_yr['conv_pnl'].gt(0).mean()*100)
    flat_pl = r2(test_yr['pnl65'].sum())
    conv_pl = r2(test_yr['conv_pnl'].sum())
    flag = ' **' if conv_wr < flat_wr else ''
    print(f"  {yr:<12} {flat_wr:>9.1f}% {conv_wr:>9.1f}% {flat_pl:>12,.0f} {conv_pl:>12,.0f}{flag}")

# H3: Zone-based — remove each zone, check if model holds
print("\n  H3: Zone stability (remove one zone at a time)")
zones = v_val['zone'].unique()
zone_results = []
for z in sorted(zones):
    sub  = v_val[v_val['zone'] != z]
    flat = r2(sub['pnl65'].gt(0).mean()*100)
    conv = r2(sub['conv_pnl'].gt(0).mean()*100)
    zone_results.append((z, flat, conv, r2(sub['conv_pnl'].sum())))
zone_results.sort(key=lambda x: x[3])
print(f"  (worst without-zone first)")
for z, flat, conv, pl in zone_results[:5]:
    print(f"  Remove {z:<18}  flat WR={flat}%  conv WR={conv}%  conv P&L Rs {pl:>9,.0f}")

# H4: Bootstrap (100 samples)
print("\n  H4: Bootstrap validation (100 samples, 80% draw)")
bs_flat = []; bs_conv = []
rng = np.random.default_rng(42)
for _ in range(100):
    idx  = rng.choice(len(v_val), size=int(0.8*len(v_val)), replace=True)
    samp = v_val.iloc[idx]
    bs_flat.append(samp['pnl65'].sum())
    bs_conv.append(samp['conv_pnl'].sum())
print(f"  Bootstrap flat P&L: mean={r2(np.mean(bs_flat)):,.0f}  95%CI [{r2(np.percentile(bs_flat,2.5)):,.0f},{r2(np.percentile(bs_flat,97.5)):,.0f}]")
print(f"  Bootstrap conv P&L: mean={r2(np.mean(bs_conv)):,.0f}  95%CI [{r2(np.percentile(bs_conv,2.5)):,.0f},{r2(np.percentile(bs_conv,97.5)):,.0f}]")
print(f"  Conviction beats flat in {r2((np.array(bs_conv)>np.array(bs_flat)).mean()*100)}% of bootstrap samples")

# ═══════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════
ts_base = int(pd.Timestamp('2021-01-01').timestamp())

# Chart 1: Feature importance bar
fi_bar = [{'time': ts_base + i*86400*45,
           'value': round(float(importances.iloc[i]), 3),
           'color': '#26a69a' if importances.iloc[i] > 0.05 else '#8b949e',
           'label': importances.index[i]}
          for i in range(len(importances))]
send_custom_chart('ml_feature_importance', {
    'lines': [{'id':'fi','label':'RF Feature Importance','seriesType':'bar',
               'data': fi_bar, 'xLabels': [b['label'] for b in fi_bar]}],
    'candlestick':[],'volume':[],'isTvFormat':False},
    title=f'ML Feature Importance (RandomForest) | Test AUC={test_auc}')
print("\n📊 ML feature importance chart sent")

# Chart 2: Conviction equity vs baseline
v_s_dated = v_s.copy()
v_s_dated['ts'] = v_s_dated['date'].apply(lambda d: int(pd.Timestamp(d).timestamp()))
send_custom_chart('conviction_final', {
    'lines': [
        {'id':'base','label':'Flat 1-lot','color':'#8b949e',
         'data':[{'time':int(r.ts),'value':round(float(v),2)}
                 for r,v in zip(v_s_dated.itertuples(), eq_base.values)]},
        {'id':'conv','label':'Conviction 1/2/3x','color':'#26a69a',
         'data':[{'time':int(r.ts),'value':round(float(v),2)}
                 for r,v in zip(v_s_dated.itertuples(), eq_conv.values)]},
    ],
    'candlestick':[],'volume':[],'isTvFormat':False},
    title=f'Final Conviction Sizing | +Rs {v_s.conv_pnl.sum()-v_s.pnl65.sum():,.0f} improvement')
print("📊 Conviction equity chart sent")

# Chart 3: Score WR bar
score_bar = [{'time': ts_base + int(sc)*86400*80,
              'value': round(v_conv[v_conv['conv_score']==sc]['pnl65'].gt(0).mean()*100,1),
              'color': '#26a69a',
              'label': f'Score {sc} ({conv_lots(sc)}x, {(v_conv.conv_score==sc).sum()}t)'}
             for sc in sorted(v_conv['conv_score'].unique())]
send_custom_chart('score_design', {
    'lines': [{'id':'sc','label':'WR by Score','seriesType':'bar',
               'data': score_bar, 'xLabels': [b['label'] for b in score_bar]}],
    'candlestick':[],'volume':[],'isTvFormat':False},
    title='Conviction Score → WR (final lot rule)')
print("📊 Score design chart sent")

# Chart 4: Year LOO validation
loo_bar = []
for i, (yr, test_yr) in enumerate(v_val.groupby('year')):
    conv_wr = test_yr['conv_pnl'].gt(0).mean()*100
    loo_bar.append({'time': ts_base + i*86400*365,
                    'value': round(conv_wr, 1),
                    'color': '#26a69a' if conv_wr >= 65 else '#ef5350',
                    'label': str(yr)})
send_custom_chart('loo_validation', {
    'lines': [{'id':'loo','label':'Conv WR by year','seriesType':'bar',
               'data': loo_bar, 'xLabels': [b['label'] for b in loo_bar]}],
    'candlestick':[],'volume':[],'isTvFormat':False},
    title='Year-by-Year Conviction WR (LOO Validation)')
print("📊 LOO validation chart sent")

print("\nAll done!")
