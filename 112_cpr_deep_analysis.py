"""
112_cpr_deep_analysis.py — CPR deep dive analysis → Excel export
================================================================
For each BASE strategy trade enrich with:
  - CPR width (TC-BC), narrow/wide classification
  - Standard pivot R1-R4, S1-S4
  - Spot OHLC on trade day
  - Spot MAE / MFE from entry (spot move against/with our direction)
  - Option MAE (max premium rise from entry = max risk)
  - Option MFE (max premium drop from entry = max profit)
  - First pivot level hit after entry (R1/R2/S1/S2...)
  - What happened at that level (break / rejection)
  - Entry time, exit time, time-in-trade
  - Confluence signals at entry

Output: Excel with 4 sheets
  1. full_data     — one row per trade, all columns
  2. by_cpr_width  — stats grouped by narrow/wide CPR
  3. by_first_level — stats grouped by first pivot level hit
  4. by_exit_reason — stats grouped by exit type
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates

OUT_DIR    = 'data/20260430'
STRIKE_INT = 50
LOT_SIZE   = 75
SCALE      = 65 / 75
EOD_EXIT   = '15:20:00'

def r2(v): return round(float(v), 2)
def get_atm(spot): return int(round(spot / STRIKE_INT) * STRIKE_INT)


# ── Pre-build previous day OHLC map ───────────────────────────────────────────
print("Loading daily OHLC...")
t0 = time.time()
all_dates = list_trading_dates()
ohlc_rows = []
for d in all_dates:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
    if len(day) < 2: continue
    ohlc_rows.append({
        'date': d,
        'o': r2(day.iloc[0]['price']),
        'h': r2(day['price'].max()),
        'l': r2(day['price'].min()),
        'c': r2(day.iloc[-1]['price']),
    })
df_ohlc = pd.DataFrame(ohlc_rows)
df_ohlc['pdh'] = df_ohlc['h'].shift(1)
df_ohlc['pdl'] = df_ohlc['l'].shift(1)
df_ohlc['pdc'] = df_ohlc['c'].shift(1)
ohlc_map = {row['date']: row for _, row in df_ohlc.iterrows()}
print(f"  {len(ohlc_map)} days in {time.time()-t0:.0f}s")


def compute_pivots(pdh, pdl, pdc):
    """Standard pivots + CPR."""
    p   = r2((pdh + pdl + pdc) / 3)
    bc  = r2((pdh + pdl) / 2)
    tc  = r2(2 * p - bc)
    r1  = r2(2 * p - pdl)
    r2_ = r2(p + (pdh - pdl))
    r3  = r2(pdh + 2 * (p - pdl))
    r4  = r2(pdh + 3 * (p - pdl))
    s1  = r2(2 * p - pdh)
    s2_ = r2(p - (pdh - pdl))
    s3  = r2(pdl - 2 * (pdh - p))
    s4  = r2(pdl - 3 * (pdh - p))
    return dict(pivot=p, bc=bc, tc=tc,
                r1=r1, r2=r2_, r3=r3, r4=r4,
                s1=s1, s2=s2_, s3=s3, s4=s4)


def first_level_hit(spot_tks, entry_time, levels_dict, signal):
    """
    After entry_time, which named level (R1/R2/S1/S2..) does spot touch first?
    Returns (level_name, touch_time, rejection: bool)
    """
    scan = spot_tks[spot_tks['time'] >= entry_time].reset_index(drop=True)
    if scan.empty: return None, None, None

    # Build ordered list of levels relevant to direction
    # CE signal = bearish = watch support levels below + resistance above
    named = [(k, v) for k, v in levels_dict.items()
             if k not in ('pivot', 'bc', 'tc') and v is not None]
    named = sorted(named, key=lambda x: x[1])  # sorted by price

    tol = 0.002  # 0.2% touch tolerance
    first_lvl = None; first_time = None
    for _, row in scan.iterrows():
        p = row['price']; t = row['time']
        for name, lv in named:
            if abs(p - lv) / p <= tol:
                first_lvl = name; first_time = t
                break
        if first_lvl: break

    if first_lvl is None: return None, None, None

    # Rejection: after touch, does price reverse ≥0.15%?
    touch_idx = scan[scan['time'] >= first_time].index[0] if first_time else None
    rejection = False
    if touch_idx is not None:
        after = scan.loc[touch_idx: touch_idx+30]['price'].values
        if len(after) > 5:
            if signal == 'CE':
                # sell CE = bearish. Rejection at resistance = price drops after touch
                rejection = (after[-1] < after[0] * (1 - 0.0015))
            else:
                rejection = (after[-1] > after[0] * (1 + 0.0015))
    return first_lvl, first_time, rejection


def _worker(args):
    trade, ohlc_row = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    date_str   = trade['date_str']
    entry_time = trade['entry_time']
    opt        = trade['opt']         # CE or PE
    ep         = trade['entry_price']
    xp         = trade['exit_price']
    exit_reason= trade['exit_reason']
    pnl        = trade['pnl']

    try:
        # ── Pivots ────────────────────────────────────────────────────────────
        pvts = {}
        cpr_width = None; cpr_class = None
        if ohlc_row is not None:
            pdh = ohlc_row['pdh']; pdl = ohlc_row['pdl']; pdc = ohlc_row['pdc']
            if not (np.isnan(pdh) or np.isnan(pdl) or np.isnan(pdc)):
                pvts = compute_pivots(pdh, pdl, pdc)
                cpr_width = r2(abs(pvts['tc'] - pvts['bc']))

        # ── Spot data ─────────────────────────────────────────────────────────
        spot_all = load_spot_data(date_str, 'NIFTY')
        if spot_all is None: return None
        spot_day = spot_all[(spot_all['time'] >= '09:15:00') & (spot_all['time'] <= '15:30:00')]
        if spot_day.empty: return None

        spot_at = spot_day[spot_day['time'] >= entry_time]
        if spot_at.empty: return None
        spot_entry = r2(spot_at.iloc[0]['price'])

        # Spot OHLC
        spot_o = r2(spot_day.iloc[0]['price'])
        spot_h = r2(spot_day['price'].max())
        spot_l = r2(spot_day['price'].min())
        spot_c = r2(spot_day.iloc[-1]['price'])

        # Spot MAE/MFE from entry (in spot points)
        after_entry = spot_day[spot_day['time'] >= entry_time]['price'].values
        if opt == 'CE':
            # Selling CE = bearish. Adverse = spot going UP, Favorable = spot going DOWN
            spot_mfe = r2(spot_entry - after_entry.min())   # max favorable (spot drops)
            spot_mae = r2(after_entry.max() - spot_entry)   # max adverse (spot rises)
        else:
            spot_mfe = r2(after_entry.max() - spot_entry)
            spot_mae = r2(spot_entry - after_entry.min())

        # ── Option data ───────────────────────────────────────────────────────
        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return None
        atm   = get_atm(spot_entry)
        instr = f'NIFTY{expiries[0]}{atm}{opt}'

        opt_tks = load_tick_data(date_str, instr, entry_time)
        if opt_tks is None or opt_tks.empty: return None
        opt_scan = opt_tks[opt_tks['time'] >= entry_time]
        if opt_scan.empty: return None

        opt_prices = opt_scan['price'].values
        opt_times  = opt_scan['time'].values

        # MAE = max price rise (bad for seller)
        opt_mae = r2(opt_prices.max() - ep)
        # MFE = max price drop (good for seller)
        opt_mfe = r2(ep - opt_prices.min())

        # Exit time
        exit_time = None
        for i, (t, p) in enumerate(zip(opt_times, opt_prices)):
            if t >= EOD_EXIT:
                exit_time = t; break
            pct_drop = (ep - p) / ep
            pct_rise = (p - ep) / ep
            if pct_drop >= 0.30:
                exit_time = t; break
            if pct_rise >= 1.00:
                exit_time = t; break
        if exit_time is None and len(opt_times): exit_time = opt_times[-1]

        # Time in trade (minutes)
        def t2m(tstr):
            h, m, s = map(int, tstr.split(':'))
            return h*60 + m + s/60
        time_in_trade = r2(t2m(exit_time) - t2m(entry_time)) if exit_time else None

        # ── First pivot level hit ─────────────────────────────────────────────
        fl_name, fl_time, fl_rej = first_level_hit(spot_day, entry_time, pvts, opt)

        # ── CPR class (narrow/wide) ───────────────────────────────────────────
        # Will be filled after all rows are collected (median-based)

        # ── Confluence signals ────────────────────────────────────────────────
        conf_signals = []
        if pvts:
            if opt == 'CE':
                if spot_o < pvts.get('bc', 0): conf_signals.append('open_below_BC')
                if spot_entry < pvts.get('pivot', 0): conf_signals.append('entry_below_pivot')
                if spot_entry < pvts.get('s1', 0): conf_signals.append('entry_below_S1')
            else:
                if spot_o > pvts.get('tc', 0): conf_signals.append('open_above_TC')
                if spot_entry > pvts.get('pivot', 0): conf_signals.append('entry_above_pivot')
                if spot_entry > pvts.get('r1', 0): conf_signals.append('entry_above_R1')

        return {
            'date': date_str,
            'opt': opt,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'time_in_trade_min': time_in_trade,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'win': pnl > 0,
            # Spot OHLC
            'spot_open': spot_o, 'spot_high': spot_h,
            'spot_low': spot_l,  'spot_close': spot_c,
            'spot_entry': spot_entry,
            'spot_mae': spot_mae, 'spot_mfe': spot_mfe,
            # Option MAE/MFE
            'opt_strike': atm, 'opt_ep': ep, 'opt_xp': xp,
            'opt_mae': opt_mae, 'opt_mfe': opt_mfe,
            # Pivots
            'cpr_width': cpr_width,
            'pivot': pvts.get('pivot'), 'bc': pvts.get('bc'), 'tc': pvts.get('tc'),
            'r1': pvts.get('r1'), 'r2': pvts.get('r2'),
            'r3': pvts.get('r3'), 'r4': pvts.get('r4'),
            's1': pvts.get('s1'), 's2': pvts.get('s2'),
            's3': pvts.get('s3'), 's4': pvts.get('s4'),
            # First level hit
            'first_level': fl_name,
            'first_level_time': fl_time,
            'first_level_rejection': fl_rej,
            # Confluence
            'confluence': '|'.join(conf_signals) if conf_signals else '',
            'conf_count': len(conf_signals),
        }
    except Exception as e:
        return None


def main():
    # Load base trades
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date_str'] = pd.to_datetime(base['date'].astype(str), format='mixed').dt.strftime('%Y%m%d')
    base = base.rename(columns={'pnl_conv': 'pnl'})

    trade_list = []
    for _, row in base.iterrows():
        d = row['date_str']
        trade_list.append({
            'date_str':    d,
            'entry_time':  row['entry_time'],
            'opt':         row['opt'],
            'entry_price': row['entry_price'],
            'exit_price':  row['exit_price'],
            'exit_reason': row['exit_reason'],
            'pnl':         row['pnl'],
        })

    args_list = [(t, ohlc_map.get(t['date_str'])) for t in trade_list]
    print(f"\nEnriching {len(args_list)} base trades (parallel)...")
    t0 = time.time()
    n_workers = min(16, cpu_count() or 4)
    with Pool(processes=n_workers) as pool:
        results = pool.map(_worker, args_list)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    df = pd.DataFrame([r for r in results if r is not None])
    if df.empty:
        print("No results."); return

    # ── Classify CPR width ────────────────────────────────────────────────────
    cpr_med = df['cpr_width'].median()
    df['cpr_class'] = df['cpr_width'].apply(
        lambda x: 'narrow' if (x is not None and x < cpr_med) else 'wide')
    print(f"  CPR width median: {cpr_med:.1f} pts | narrow<={cpr_med:.0f} | wide>{cpr_med:.0f}")

    # ── Print summary ─────────────────────────────────────────────────────────
    def stats(sub, label):
        if sub.empty: return
        wr  = sub['win'].mean() * 100
        tp  = sub['pnl'].sum()
        ap  = sub['pnl'].mean()
        mae = sub['opt_mae'].mean()
        mfe = sub['opt_mfe'].mean()
        print(f"  {label:<30} | {len(sub):>4}t | WR {wr:>5.1f}% | "
              f"Rs.{tp:>9,.0f} | Avg Rs.{ap:>5,.0f} | MAE {mae:>5.0f} | MFE {mfe:>5.0f}")

    print(f"\n{'='*90}")
    print(f"  BASE STRATEGY DEEP ANALYSIS — CPR / Pivots / MAE / MFE")
    print(f"{'='*90}")
    print(f"  {'Variant':<30} | {'T':>4} | {'WR':>7} | {'Total P&L':>11} | {'Avg':>8} | {'MAE':>7} | {'MFE':>7}")
    print(f"  {'-'*88}")
    stats(df, 'All trades')
    stats(df[df['opt']=='CE'], 'CE trades')
    stats(df[df['opt']=='PE'], 'PE trades')
    print(f"  {'-'*88}")
    stats(df[df['cpr_class']=='narrow'], 'Narrow CPR')
    stats(df[df['cpr_class']=='wide'],   'Wide CPR')
    print(f"  {'-'*88}")
    stats(df[df['cpr_class']=='narrow'][df['opt']=='CE'], 'Narrow CPR + CE')
    stats(df[df['cpr_class']=='narrow'][df['opt']=='PE'], 'Narrow CPR + PE')
    stats(df[df['cpr_class']=='wide'][df['opt']=='CE'],   'Wide CPR + CE')
    stats(df[df['cpr_class']=='wide'][df['opt']=='PE'],   'Wide CPR + PE')
    print(f"  {'-'*88}")
    for fl, g in df.groupby('first_level'):
        stats(g, f'First hit: {fl}')
    print(f"  {'-'*88}")
    for er, g in df.groupby('exit_reason'):
        stats(g, f'Exit: {er}')

    # MAE/MFE distribution
    print(f"\n  Option MAE/MFE distribution (wins vs losses):")
    wins  = df[df['win']==True]
    losses= df[df['win']==False]
    print(f"    Wins   ({len(wins)}):  MAE avg {wins['opt_mae'].mean():.1f} | MFE avg {wins['opt_mfe'].mean():.1f} | "
          f"MAE<ep: {(wins['opt_mae']<wins['opt_ep']).mean()*100:.0f}%")
    print(f"    Losses ({len(losses)}): MAE avg {losses['opt_mae'].mean():.1f} | MFE avg {losses['opt_mfe'].mean():.1f} | "
          f"MAE<ep: {(losses['opt_mae']<losses['opt_ep']).mean()*100:.0f}%")

    # MAE filter test: what if we skip trades where MAE > X*ep before reaching 10% profit?
    print(f"\n  MAE/EP ratio distribution:")
    df['mae_ep_ratio'] = (df['opt_mae'] / df['opt_ep']).round(2)
    for thr in [0.25, 0.50, 0.75, 1.00]:
        sub = df[df['mae_ep_ratio'] <= thr]
        if len(sub):
            print(f"    MAE <= {int(thr*100)}% of EP → {len(sub)}t | WR {sub['win'].mean()*100:.1f}% | "
                  f"Avg Rs.{sub['pnl'].mean():.0f}")

    # ── Build summary sheets ───────────────────────────────────────────────────
    def grp_stats(g):
        return pd.Series({
            'trades':   len(g),
            'win_rate': round(g['win'].mean()*100, 1),
            'total_pnl':round(g['pnl'].sum(), 0),
            'avg_pnl':  round(g['pnl'].mean(), 0),
            'opt_mae_avg': round(g['opt_mae'].mean(), 1),
            'opt_mfe_avg': round(g['opt_mfe'].mean(), 1),
            'spot_mae_avg':round(g['spot_mae'].mean(), 1),
            'spot_mfe_avg':round(g['spot_mfe'].mean(), 1),
            'time_in_trade_avg': round(g['time_in_trade_min'].mean(), 0),
        })

    by_cpr    = df.groupby(['cpr_class','opt']).apply(grp_stats).reset_index()
    by_level  = df.groupby('first_level').apply(grp_stats).reset_index()
    by_exit   = df.groupby('exit_reason').apply(grp_stats).reset_index()
    by_year   = df.groupby(df['date'].str[:4]).apply(grp_stats).reset_index()

    # ── Export to Excel ────────────────────────────────────────────────────────
    excel_path = f'{OUT_DIR}/112_cpr_deep_analysis.xlsx'
    col_order = [
        'date','opt','entry_time','exit_time','time_in_trade_min','exit_reason','pnl','win',
        'spot_open','spot_high','spot_low','spot_close','spot_entry','spot_mae','spot_mfe',
        'opt_strike','opt_ep','opt_xp','opt_mae','opt_mfe','mae_ep_ratio',
        'cpr_width','cpr_class',
        'pivot','bc','tc','r1','r2','r3','r4','s1','s2','s3','s4',
        'first_level','first_level_time','first_level_rejection',
        'conf_count','confluence',
    ]
    col_order = [c for c in col_order if c in df.columns]

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df[col_order].to_excel(writer, sheet_name='full_data',    index=False)
        by_cpr.to_excel(writer,        sheet_name='by_cpr_width',  index=False)
        by_level.to_excel(writer,      sheet_name='by_first_level',index=False)
        by_exit.to_excel(writer,       sheet_name='by_exit_reason',index=False)
        by_year.to_excel(writer,       sheet_name='by_year',       index=False)

    print(f"\n  Saved → {excel_path}")
    print(f"  Sheets: full_data | by_cpr_width | by_first_level | by_exit_reason | by_year")
    print("\nDone.")

if __name__ == '__main__':
    main()
