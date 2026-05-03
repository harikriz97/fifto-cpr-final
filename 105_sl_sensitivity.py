"""
105_sl_sensitivity.py — Hard SL sensitivity: 60% vs 70% vs 80% vs 100% (baseline)
===================================================================================
Tests tighter hard SL on CRT+MRC blank day strategy (ATM 30%).
Goal: reduce max drawdown without killing total P&L.

Hard SL = initial stop loss as % above entry price:
  100% → option must double before stopping out  (current)
   80% → option must go +80% before stopping out
   70% → option must go +70% before stopping out
   60% → option must go +60% before stopping out
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, f'{os.path.expanduser("~")}/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import pandas as pd
from multiprocessing import Pool, cpu_count
from datetime import datetime
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates
from plot_util import send_custom_chart

OUT_DIR    = 'data/20260430'
LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
EOD_EXIT   = '15:20:00'
TGT_PCT    = 0.30

def r2(v): return round(float(v), 2)
def get_atm(spot): return int(round(spot / STRIKE_INT) * STRIKE_INT)

def simulate_sell(date_str, instrument, entry_time, hard_sl_pct):
    tks = load_tick_data(date_str, instrument, entry_time)
    if tks is None or tks.empty: return None
    tks = tks[tks['time'] >= entry_time].reset_index(drop=True)
    if tks.empty: return None
    ep = r2(tks.iloc[0]['price'])
    if ep <= 0: return None
    tgt = r2(ep * (1 - TGT_PCT))
    hsl = r2(ep * (1 + hard_sl_pct))  # e.g. 60% → ep*1.60
    sl  = hsl
    md  = 0.0
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


def _worker(args):
    date_str, signal_type, signal_dir, entry_time, hard_sl_pct = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        spot = load_spot_data(date_str, 'NIFTY')
        if spot is None: return None
        spot_at = spot[spot['time'] >= entry_time[:8]]
        if spot_at.empty: return None
        spot_ref = spot_at.iloc[0]['price']

        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return None
        strike = get_atm(spot_ref)
        instr  = f'NIFTY{expiries[0]}{strike}{signal_dir}'

        res = simulate_sell(date_str, instr, entry_time, hard_sl_pct)
        if res is None: return None
        pnl, reason, ep, xp = res
        return {'date': date_str, 'signal_type': signal_type, 'pnl': pnl,
                'win': pnl > 0, 'exit_reason': reason, 'ep': ep}
    except Exception as e:
        return None


def run_for_sl(args_base, sl_pct):
    args_list = [(d, st, sig, et, sl_pct) for d, st, sig, et in args_base]
    n_workers = min(16, cpu_count() or 4)
    with Pool(processes=n_workers) as pool:
        results = pool.map(_worker, args_list)
    return [r for r in results if r is not None]


# ── Load signal lookup ─────────────────────────────────────────────────────────
print("Loading signal lookup tables...")
crt_raw   = pd.read_csv(f'{OUT_DIR}/91_crt_ltf_D.csv')
crt_blank = crt_raw[crt_raw['is_blank']==True].copy()
crt_blank['date'] = crt_blank['date'].astype(str)
crt_sig   = {row['date']: row['entry_time'] for _, row in crt_blank.iterrows()}

mrc_raw   = pd.read_csv(f'{OUT_DIR}/100_mrc_trades.csv')
mrc_blank = mrc_raw[mrc_raw['is_blank']==True].copy()
mrc_blank['date'] = mrc_blank['date'].astype(str)
crt_dates  = set(crt_sig.keys())
mrc_unique = mrc_blank[~mrc_blank['date'].isin(crt_dates)].copy()
mrc_sig    = {row['date']: (row['entry_time'], row['signal'])
              for _, row in mrc_unique.iterrows()}

all_dates = list_trading_dates()

# build base args (without SL param)
args_base = []
for d in all_dates:
    if d in crt_sig:
        args_base.append((d, 'CRT', 'CE', crt_sig[d]))
    elif d in mrc_sig:
        et, sig = mrc_sig[d]
        args_base.append((d, 'MRC', sig, et))

print(f"Signal dates: {len(args_base)}")

# ── Load base strategy for combined stats ──────────────────────────────────────
base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base['date'] = pd.to_datetime(base['date'].astype(str), format='mixed')
base_daily = base.groupby('date')['pnl_conv'].sum().reset_index()
base_daily.columns = ['date', 'base_pnl']

SL_LEVELS = [1.00, 0.80, 0.70, 0.60]
summary_rows = []
eq_lines = []

for sl in SL_LEVELS:
    label = f"SL {int(sl*100)}%"
    t0 = time.time()
    trades = run_for_sl(args_base, sl)
    elapsed = time.time() - t0

    df = pd.DataFrame(trades).sort_values('date')
    df['date_dt'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')

    # daily P&L
    blank_daily = df.groupby('date_dt')['pnl'].sum().reset_index()
    blank_daily.columns = ['date', 'blank_pnl']

    all_dt = pd.DataFrame({'date': sorted(set(base_daily['date']) | set(blank_daily['date']))})
    m = all_dt.merge(base_daily, on='date', how='left').merge(blank_daily, on='date', how='left')
    m['base_pnl']  = m['base_pnl'].fillna(0)
    m['blank_pnl'] = m['blank_pnl'].fillna(0)
    m['comb_pnl']  = m['base_pnl'] + m['blank_pnl']
    m['comb_eq']   = m['comb_pnl'].cumsum()

    total      = len(df)
    wins       = df['win'].sum()
    wr         = round(wins / total * 100, 2)
    total_pnl  = round(df['pnl'].sum(), 2)
    avg_pnl    = round(df['pnl'].mean(), 2)
    hard_sl_ct = (df['exit_reason'] == 'hard_sl').sum()
    comb_total = round(m['comb_eq'].iloc[-1], 0)
    comb_dd    = round((m['comb_eq'] - m['comb_eq'].cummax()).min(), 0)
    worst_day  = round(df['pnl'].min(), 0)

    summary_rows.append({
        'sl_pct': int(sl*100),
        'trades': total,
        'wr': wr,
        'blank_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'hard_sl_count': hard_sl_ct,
        'combined_total': comb_total,
        'combined_dd': comb_dd,
        'worst_day': worst_day,
    })

    eq_lines.append({
        'label': label,
        'dates': m['date'].tolist(),
        'eq':    m['comb_eq'].tolist(),
    })

    print(f"  SL {int(sl*100)}%: {total}t | WR {wr}% | Blank Rs.{total_pnl:,.0f} | "
          f"Avg Rs.{avg_pnl:.0f} | HardSL {hard_sl_ct} | "
          f"Combined Rs.{comb_total:,.0f} | DD Rs.{comb_dd:,.0f} | "
          f"Worst Rs.{worst_day:,.0f} | {elapsed:.1f}s")

# ── Print comparison table ─────────────────────────────────────────────────────
df_s = pd.DataFrame(summary_rows)
print(f"\n{'='*80}")
print(f"  HARD SL SENSITIVITY — CRT+MRC Blank Days (ATM 30%)")
print(f"{'='*80}")
print(f"  {'SL':>4} | {'Trades':>6} | {'WR':>6} | {'Blank P&L':>11} | {'Avg':>7} | "
      f"{'HardSL':>6} | {'Combined':>12} | {'DD':>10} | {'Worst Day':>10}")
print(f"  {'-'*78}")
for _, r in df_s.iterrows():
    print(f"  {r['sl_pct']:>3}% | {r['trades']:>6} | {r['wr']:>5.1f}% | "
          f"Rs.{r['blank_pnl']:>8,.0f} | Rs.{r['avg_pnl']:>4,.0f} | "
          f"{r['hard_sl_count']:>6} | Rs.{r['combined_total']:>9,.0f} | "
          f"Rs.{r['combined_dd']:>7,.0f} | Rs.{r['worst_day']:>7,.0f}")

df_s.to_csv(f'{OUT_DIR}/105_sl_sensitivity.csv', index=False)

# ── Chart: 4 combined equity lines ────────────────────────────────────────────
colors = ['#9e9e9e', '#0ea5e9', '#f59e0b', '#26a69a']  # 100%, 80%, 70%, 60%

lines = []
for i, eq in enumerate(eq_lines):
    lines.append({
        "id": f"sl_{eq['label'].replace(' ','')}",
        "label": eq['label'],
        "color": colors[i],
        "data": [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                 for d, v in zip(eq['dates'], eq['eq']) if pd.notna(v)],
        "seriesType": "line",
    })

tv_json = {"isTvFormat": False, "candlestick": [], "volume": [], "lines": lines}

best = df_s.loc[df_s['combined_dd'].abs().idxmin()]
send_custom_chart("105_sl_sensitivity", tv_json,
                  title=f"Hard SL Sensitivity — ATM 30% | Best DD: SL {int(best['sl_pct'])}% Rs.{best['combined_dd']:,.0f}")

print(f"\n  Saved → {OUT_DIR}/105_sl_sensitivity.csv")
print("\nDone.")
