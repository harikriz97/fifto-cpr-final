"""
120_improvements.py — Targeted improvements to base strategy
=============================================================
Track A: Late Entry Base (9:46:02) with IB direction filter
  Same signal days as base. Entry moved to 9:46:02 after IB complete.
  Filter:
    PE sell → skip if IB expands DOWN (market falling → PE risky to sell)
    CE sell → skip if IB expands UP   (market rising  → CE risky to sell)
  Directly targets: normal_down + above_tc (102t | WR 38.2% | avg -₹3,014)

Track B: Target sweep
  Re-simulate all base signal days with targets: 20%, 25%, 30%, 35%, 40%
  Keep same trailing SL logic — find optimal target.

Track C: Partial booking
  Book 50% at first_target (15% / 20%), trail remaining 50% to full_target (30%)
  3 variants:
    C1: book 50% @ 15%, trail rest (lock-in SL at entry) → full target 30%
    C2: book 50% @ 20%, trail rest → full target 30%
    C3: book 75% @ 15%, trail 25% → full target 30%

Track D: CPR-width sizing
  Narrow CPR (< 50pts): 0.5x lot   (uncertain direction)
  Medium CPR (50-150):  1.0x lot   (normal)
  Wide CPR  (> 150pts): 1.5x lot   (strong trend likely)
  Uses existing base trades — no resimulation needed.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, f'{os.path.expanduser("~")}/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from datetime import datetime
from my_util import load_spot_data, load_tick_data, list_expiry_dates
from plot_util import send_custom_chart

OUT_DIR    = 'data/20260430'
LOT_SIZE   = 75
SCALE      = 65 / 75
STRIKE_INT = 50
IB_END     = '09:45:00'
LATE_ET    = '09:46:02'
EOD_EXIT   = '15:20:00'
BASE_TGT   = 0.30

TARGET_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40]

def r2(v): return round(float(v), 2)
def get_atm(s): return int(round(s / STRIKE_INT) * STRIKE_INT)


def simulate_sell_full(tks_arr, ep, tgt_pct, partial_pct=None, partial_tgt_pct=None):
    """
    Full tick simulation. Returns (pnl, reason, exit_price).
    If partial_pct is set: book partial_pct at partial_tgt_pct,
    trail rest to tgt_pct (lock-in SL at entry after partial).
    """
    tgt  = r2(ep * (1 - tgt_pct))
    hsl  = r2(ep * 2.0); sl = hsl; md = 0.0
    partial_done = False
    partial_pnl  = 0.0
    rem_pct      = 1.0

    if partial_pct is not None:
        ptgt = r2(ep * (1 - partial_tgt_pct))
        rem_pct = 1.0 - partial_pct

    ps = tks_arr['price'].values
    ts = tks_arr['time'].values

    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= EOD_EXIT:
            total = (partial_pnl + rem_pct * r2((ep - p) * LOT_SIZE * SCALE))
            return r2(total), 'eod', r2(p)

        d = (ep - p) / ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep * (1 - md * 0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep * 0.80))
        elif md >= 0.25: sl = min(sl, ep)

        # Partial booking
        if partial_pct is not None and not partial_done:
            if p <= ptgt:
                partial_pnl = partial_pct * r2((ep - p) * LOT_SIZE * SCALE)
                partial_done = True
                sl = ep   # lock-in SL at entry for remaining
                continue

        # Full target
        if p <= tgt:
            total = partial_pnl + rem_pct * r2((ep - p) * LOT_SIZE * SCALE)
            return r2(total), 'target', r2(p)
        if p >= sl:
            reason = 'lockin_sl' if sl < hsl else 'hard_sl'
            total  = partial_pnl + rem_pct * r2((ep - p) * LOT_SIZE * SCALE)
            return r2(total), reason, r2(p)

    total = partial_pnl + rem_pct * r2((ep - ps[-1]) * LOT_SIZE * SCALE)
    return r2(total), 'eod', r2(ps[-1])


# ── TRACK A: Late entry with IB filter ────────────────────────────────────────
def _worker_late_entry(args):
    date_str, opt, expiry, atm_original = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return None
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')].copy()
        if len(day) < 30: return None

        ib = day[day['time'] <= IB_END]
        if len(ib) < 5: return None
        ib_h = ib['price'].max(); ib_l = ib['price'].min()

        ib_exp_up   = day[(day['time'] > IB_END) & (day['time'] < LATE_ET)]['price'].max() > ib_h
        ib_exp_down = day[(day['time'] > IB_END) & (day['time'] < LATE_ET)]['price'].min() < ib_l

        # Filter: skip if IB direction contradicts the trade
        if opt == 'PE' and ib_exp_down and not ib_exp_up:
            return {'filtered': True, 'date': date_str, 'opt': opt}
        if opt == 'CE' and ib_exp_up and not ib_exp_down:
            return {'filtered': True, 'date': date_str, 'opt': opt}

        # Get expiry + ATM at entry time
        expiries = list_expiry_dates(date_str, index_name='NIFTY')
        if not expiries: return None
        spot_at = day[day['time'] >= LATE_ET]
        if spot_at.empty: return None
        atm = get_atm(spot_at.iloc[0]['price'])
        instr = f'NIFTY{expiries[0]}{atm}{opt}'

        opt_tks = load_tick_data(date_str, instr, LATE_ET)
        if opt_tks is None or opt_tks.empty: return None
        tks_e = opt_tks[opt_tks['time'] >= LATE_ET].reset_index(drop=True)
        if tks_e.empty: return None
        ep = r2(tks_e.iloc[0]['price'])
        if ep <= 0: return None

        pnl, reason, xp = simulate_sell_full(tks_e, ep, BASE_TGT)
        return {'filtered': False, 'date': date_str, 'opt': opt,
                'ep': ep, 'xp': xp, 'pnl': pnl, 'win': pnl > 0,
                'exit_reason': reason, 'ib_exp_up': ib_exp_up, 'ib_exp_down': ib_exp_down}
    except Exception:
        return None


# ── TRACK B: Target sweep ──────────────────────────────────────────────────────
def _worker_target_sweep(args):
    date_str, opt, entry_time, expiry, atm = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        instr = f'NIFTY{expiry}{atm}{opt}'
        opt_tks = load_tick_data(date_str, instr, entry_time)
        if opt_tks is None or opt_tks.empty: return []
        tks_e = opt_tks[opt_tks['time'] >= entry_time].reset_index(drop=True)
        if tks_e.empty: return []
        ep = r2(tks_e.iloc[0]['price'])
        if ep <= 0: return []

        results = []
        for tgt_pct in TARGET_SWEEP:
            pnl, reason, xp = simulate_sell_full(tks_e, ep, tgt_pct)
            results.append({
                'tgt_pct': round(tgt_pct * 100, 0),
                'date': date_str, 'opt': opt,
                'ep': ep, 'xp': xp, 'pnl': pnl, 'win': pnl > 0,
                'exit_reason': reason,
            })
        return results
    except Exception:
        return []


# ── TRACK C: Partial booking ───────────────────────────────────────────────────
def _worker_partial(args):
    date_str, opt, entry_time, expiry, atm = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        instr = f'NIFTY{expiry}{atm}{opt}'
        opt_tks = load_tick_data(date_str, instr, entry_time)
        if opt_tks is None or opt_tks.empty: return []
        tks_e = opt_tks[opt_tks['time'] >= entry_time].reset_index(drop=True)
        if tks_e.empty: return []
        ep = r2(tks_e.iloc[0]['price'])
        if ep <= 0: return []

        variants = [
            ('base_30', None,  None,  0.30),   # current baseline
            ('C1_50p15', 0.50, 0.15,  0.30),   # book 50% at 15%, trail to 30%
            ('C2_50p20', 0.50, 0.20,  0.30),   # book 50% at 20%, trail to 30%
            ('C3_75p15', 0.75, 0.15,  0.30),   # book 75% at 15%, trail 25% to 30%
            ('C4_50p15_25f', 0.50, 0.15, 0.25), # book 50% at 15%, trail to 25%
        ]
        results = []
        for label, pp, ptp, tgt in variants:
            pnl, reason, xp = simulate_sell_full(tks_e, ep, tgt,
                                                  partial_pct=pp, partial_tgt_pct=ptp)
            results.append({
                'variant': label, 'date': date_str, 'opt': opt,
                'ep': ep, 'xp': xp, 'pnl': pnl, 'win': pnl > 0,
                'exit_reason': reason,
            })
        return results
    except Exception:
        return []


def stats_line(sub, label):
    if sub.empty: return
    wr = sub['win'].mean()*100; tp = sub['pnl'].sum(); ap = sub['pnl'].mean()
    print(f"  {label:<38} | {len(sub):>4}t | WR {wr:>5.1f}% | "
          f"Rs.{tp:>12,.0f} | Avg Rs.{ap:>7,.0f}")


def main():
    # ── Load base trades ───────────────────────────────────────────────────────
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date_str'] = pd.to_datetime(base['date'].astype(str),
                                      format='mixed').dt.strftime('%Y%m%d')
    base = base.rename(columns={'pnl_conv': 'pnl'})

    # Load day behavior for CPR width
    beh = pd.read_excel(f'{OUT_DIR}/115_day_behavior.xlsx',
                        sheet_name='all_days_observations', dtype={'date': str})
    beh['date'] = beh['date'].astype(str)
    beh_map = {r['date']: r for _, r in beh.iterrows()}

    # Recover expiry + ATM from base trades (from instrument name or recompute)
    # Base CSV has: date, opt, entry_time, entry_price, exit_price, pnl, win, exit_reason
    # We need expiry + ATM for target sweep and partial

    from my_util import list_expiry_dates
    print("Recovering expiry/ATM for base trades...")
    expiry_cache = {}
    def get_expiry(date_str):
        if date_str not in expiry_cache:
            exps = list_expiry_dates(date_str, index_name='NIFTY')
            expiry_cache[date_str] = exps[0] if exps else None
        return expiry_cache[date_str]

    # For ATM: derive from entry_price using spot at entry_time
    # We'll use strike column if available, otherwise estimate from entry_price
    base_rows = []
    for _, row in base.iterrows():
        exp = get_expiry(row['date_str'])
        if exp is None: continue
        ep = float(row.get('entry_price', row.get('ep', 0)))
        if ep <= 0: continue
        # Estimate ATM: we need spot at entry_time. Use a proxy: round entry_price up/down.
        # Better: use strike column if present
        if 'strike' in row and pd.notna(row['strike']):
            atm = int(row['strike'])
        else:
            # Approximate: options are usually priced such that ATM CE/PE ≈ 1-3% of spot
            # We'll load spot at entry_time inline — or skip reconstruction
            atm = None
        base_rows.append({
            'date_str': row['date_str'],
            'opt': row['opt'],
            'entry_time': str(row.get('entry_time', '09:16:02')),
            'ep_base': ep,
            'expiry': exp,
            'atm': atm,
            'pnl_base': row['pnl'],
            'win_base': row['win'],
            'exit_reason_base': row.get('exit_reason',''),
        })
    base_ext = pd.DataFrame(base_rows)

    # Reconstruct ATM for rows where it's missing (load spot at entry_time)
    missing_atm = base_ext[base_ext['atm'].isna()]
    if not missing_atm.empty:
        print(f"  Reconstructing ATM for {len(missing_atm)} trades...")
        atm_map = {}
        for _, row in missing_atm.iterrows():
            try:
                sp = load_spot_data(row['date_str'], 'NIFTY')
                if sp is None: continue
                sp_at = sp[sp['time'] >= row['entry_time']]
                if sp_at.empty: continue
                atm_map[row['date_str']] = get_atm(sp_at.iloc[0]['price'])
            except Exception:
                pass
        base_ext['atm'] = base_ext.apply(
            lambda r: atm_map.get(r['date_str'], r['atm']) if pd.isna(r['atm']) else r['atm'],
            axis=1)
    base_ext = base_ext.dropna(subset=['atm', 'expiry'])
    base_ext['atm'] = base_ext['atm'].astype(int)
    print(f"  {len(base_ext)} valid base trades ready")

    # ── TRACK A: Late Entry ────────────────────────────────────────────────────
    print(f"\nTrack A: Late Entry (9:46:02) with IB filter...")
    args_a = [(r['date_str'], r['opt'], r['expiry'], r['atm'])
              for _, r in base_ext.iterrows()]
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_a = pool.map(_worker_late_entry, args_a)
    el = (datetime.now()-t0).total_seconds()

    results_a = [r for r in raw_a if r is not None]
    filtered_a  = [r for r in results_a if r.get('filtered')]
    traded_a    = [r for r in results_a if not r.get('filtered')]
    df_a = pd.DataFrame(traded_a)
    df_a_filt = pd.DataFrame(filtered_a)

    print(f"  Done in {el:.1f}s | Original: {len(base_ext)}t | "
          f"Filtered: {len(filtered_a)} | Traded: {len(traded_a)}")

    base_total = base['pnl'].sum()
    if not df_a.empty:
        stats_line(df_a, 'Late entry (all traded)')
        for opt, g in df_a.groupby('opt'):
            stats_line(g, f"  Late {opt}")
        df_a['year'] = df_a['date'].str[:4]
        print(f"\n  Year breakdown (late entry):")
        for yr, g in df_a.groupby('year'):
            print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

        # Filtered trades in original base — what did we skip?
        if not df_a_filt.empty:
            skip_base = base[base['date_str'].isin(df_a_filt['date'].astype(str))]
            print(f"\n  Skipped {len(skip_base)} base trades via filter:")
            if not skip_base.empty:
                print(f"    Original P&L: Rs.{skip_base['pnl'].sum():,.0f} | "
                      f"WR {skip_base['win'].mean()*100:.1f}% | "
                      f"Avg Rs.{skip_base['pnl'].mean():.0f}")

        # Compare: base_kept (not filtered) with late entry
        base_kept = base[~base['date_str'].isin(
            df_a_filt['date'].astype(str) if not df_a_filt.empty else [])]
        print(f"\n  COMPARISON:")
        print(f"    Base original:      Rs.{base_total:>12,.0f}  {len(base)}t")
        print(f"    Base kept (unfiltered original): Rs.{base_kept['pnl'].sum():>12,.0f}  {len(base_kept)}t")
        print(f"    Late entry traded:  Rs.{df_a['pnl'].sum():>12,.0f}  {len(df_a)}t")
        net_late = base_kept['pnl'].sum() - (base_kept['pnl'].sum() - base_total) + df_a['pnl'].sum()
        # Proper: late entry replaces base on all days
        # Days where filter skips: lose base P&L, gain nothing
        # Days where late entry trades: gain late P&L, lose base P&L
        days_late = set(df_a['date'].astype(str))
        days_filt = set(df_a_filt['date'].astype(str)) if not df_a_filt.empty else set()
        base_on_late_days = base[base['date_str'].isin(days_late)]['pnl'].sum()
        base_on_filt_days = base[base['date_str'].isin(days_filt)]['pnl'].sum()
        remaining_base = base[~base['date_str'].isin(days_late | days_filt)]['pnl'].sum()
        late_combined = remaining_base + df_a['pnl'].sum()
        print(f"\n  If we switch to late entry on all base days:")
        print(f"    Remaining base days (no change): Rs.{remaining_base:,.0f}")
        print(f"    Late-entry trades replace base:  Rs.{df_a['pnl'].sum():,.0f}  "
              f"(was Rs.{base_on_late_days:,.0f})")
        print(f"    Filtered days (0 trades):        Rs.0  "
              f"(was Rs.{base_on_filt_days:,.0f})")
        print(f"    TOTAL late-entry system:         Rs.{late_combined:,.0f}  "
              f"({(late_combined/base_total-1)*100:+.1f}% vs base)")

    # ── TRACK B: Target sweep ──────────────────────────────────────────────────
    print(f"\nTrack B: Target sweep {[int(t*100) for t in TARGET_SWEEP]}%...")
    args_b = [(r['date_str'], r['opt'], r['entry_time'], r['expiry'], r['atm'])
              for _, r in base_ext.iterrows()]
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_b = pool.map(_worker_target_sweep, args_b)
    el = (datetime.now()-t0).total_seconds()
    df_b = pd.DataFrame([t for day in raw_b for t in day])
    print(f"  {len(df_b)} rows in {el:.1f}s")

    if not df_b.empty:
        print(f"\n  {'Target':>8} | {'Trades':>6} | {'WR':>6} | {'Total P&L':>12} | {'Avg':>8} | {'vs base':>10}")
        print(f"  {'-'*62}")
        for tgt, g in df_b.groupby('tgt_pct'):
            diff = g['pnl'].sum() - base_total
            print(f"  {int(tgt):>7}% | {len(g):>6} | {g['win'].mean()*100:>5.1f}% | "
                  f"Rs.{g['pnl'].sum():>10,.0f} | Rs.{g['pnl'].mean():>7,.0f} | "
                  f"Rs.{diff:>+10,.0f}")

        # Year breakdown for best target
        best_tgt_val = df_b.groupby('tgt_pct')['pnl'].sum().idxmax()
        df_best_b = df_b[df_b['tgt_pct'] == best_tgt_val]
        print(f"\n  Best target = {int(best_tgt_val)}% | Year breakdown:")
        df_best_b['year'] = df_best_b['date'].str[:4]
        for yr, g in df_best_b.groupby('year'):
            print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

        # Exit analysis per target
        print(f"\n  Exit breakdown by target:")
        for tgt, g in df_b.groupby('tgt_pct'):
            for ex, gg in g.groupby('exit_reason'):
                pass
        tgt_exit = df_b.groupby(['tgt_pct','exit_reason']).agg(
            n=('pnl','count'), total=('pnl','sum')).reset_index()
        print(tgt_exit.to_string(index=False))

    # ── TRACK C: Partial booking ───────────────────────────────────────────────
    print(f"\nTrack C: Partial booking variants...")
    t0 = datetime.now()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        raw_c = pool.map(_worker_partial, args_b)  # same args as B
    el = (datetime.now()-t0).total_seconds()
    df_c = pd.DataFrame([t for day in raw_c for t in day])
    print(f"  {len(df_c)} rows in {el:.1f}s")

    if not df_c.empty:
        print(f"\n  {'Variant':<20} | {'Trades':>6} | {'WR':>6} | {'Total P&L':>12} | {'Avg':>8} | {'vs base':>10}")
        print(f"  {'-'*72}")
        for var, g in df_c.groupby('variant'):
            diff = g['pnl'].sum() - base_total
            print(f"  {var:<20} | {len(g):>6} | {g['win'].mean()*100:>5.1f}% | "
                  f"Rs.{g['pnl'].sum():>10,.0f} | Rs.{g['pnl'].mean():>7,.0f} | "
                  f"Rs.{diff:>+10,.0f}")

        # Year breakdown for best variant
        best_var = df_c.groupby('variant')['pnl'].sum().drop('base_30', errors='ignore').idxmax()
        df_best_c = df_c[df_c['variant'] == best_var]
        print(f"\n  Best variant = {best_var} | Year breakdown:")
        df_best_c['year'] = df_best_c['date'].str[:4]
        for yr, g in df_best_c.groupby('year'):
            print(f"    {yr}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
                  f"Rs.{g['pnl'].sum():,.0f} | Avg Rs.{g['pnl'].mean():.0f}")

    # ── TRACK D: CPR width sizing ──────────────────────────────────────────────
    print(f"\nTrack D: CPR width-based sizing (no resimulation)...")
    base_cpr = base.copy()
    base_cpr['cpr_class'] = base_cpr['date_str'].map(
        lambda d: beh_map.get(d, {}).get('cpr_class', 'medium'))

    def sized_pnl(row):
        cc = str(row.get('cpr_class', 'medium')).lower()
        if 'narrow' in cc or 'tight' in cc:
            mult = 0.5
        elif 'wide' in cc or 'broad' in cc or 'large' in cc:
            mult = 1.5
        else:
            mult = 1.0
        return row['pnl'] * mult

    base_cpr['pnl_sized'] = base_cpr.apply(sized_pnl, axis=1)
    print(f"  CPR class distribution:")
    for cc, g in base_cpr.groupby('cpr_class'):
        print(f"    {cc}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Rs.{g['pnl'].sum():,.0f} original | Rs.{g['pnl_sized'].sum():,.0f} sized")
    print(f"\n  Base original:   Rs.{base_cpr['pnl'].sum():,.0f}")
    print(f"  CPR sized:       Rs.{base_cpr['pnl_sized'].sum():,.0f}  "
          f"({(base_cpr['pnl_sized'].sum()/base_cpr['pnl'].sum()-1)*100:+.1f}%)")

    # ── COMBINED BEST ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  IMPROVEMENT SUMMARY")
    print(f"{'='*72}")
    print(f"  Base original:                Rs.{base_total:>12,.0f}  {len(base)}t")

    if not df_a.empty:
        print(f"  A: Late entry system:         Rs.{late_combined:>12,.0f}  "
              f"({(late_combined/base_total-1)*100:+.1f}%)")

    if not df_b.empty:
        best_b_pnl = df_b.groupby('tgt_pct')['pnl'].sum().max()
        print(f"  B: Best target (tgt={int(best_tgt_val)}%):   Rs.{best_b_pnl:>12,.0f}  "
              f"({(best_b_pnl/base_total-1)*100:+.1f}%)")

    if not df_c.empty:
        best_c_pnl = df_c.groupby('variant')['pnl'].sum().max()
        print(f"  C: Best partial booking:      Rs.{best_c_pnl:>12,.0f}  "
              f"({(best_c_pnl/base_total-1)*100:+.1f}%)")

    print(f"  D: CPR sized:                 Rs.{base_cpr['pnl_sized'].sum():>12,.0f}  "
          f"({(base_cpr['pnl_sized'].sum()/base_cpr['pnl'].sum()-1)*100:+.1f}%)")

    # ── Equity chart ──────────────────────────────────────────────────────────
    def daily_cum(df_, date_col='date', pnl_col='pnl', fmt='%Y%m%d'):
        if df_.empty: return pd.Series(dtype=float)
        df_ = df_.copy()
        df_['dt'] = pd.to_datetime(df_[date_col].astype(str), format=fmt)
        return df_.groupby('dt')[pnl_col].sum()

    base_d = base.copy()
    base_d['dt'] = pd.to_datetime(base_d['date'].astype(str), format='mixed')
    base_daily = base_d.groupby('dt')['pnl'].sum()

    all_idx = pd.date_range(start=base_daily.index.min(),
                            end=base_daily.index.max(), freq='B')

    def to_cum(series):
        return series.reindex(all_idx, fill_value=0).cumsum()

    def eq_pts(s):
        return [{"time": int(pd.Timestamp(d).timestamp()), "value": round(float(v), 2)}
                for d, v in s.items() if pd.notna(v)]

    lines = [{"id":"base","label":f"Base Rs.{int(base_daily.sum()):,.0f}",
               "color":"#9e9e9e","seriesType":"line","data":eq_pts(to_cum(base_daily))}]
    colors = ['#26a69a','#0ea5e9','#ab47bc','#ff9800','#66bb6a']
    ci = 0

    if not df_a.empty:
        # Reconstruct late entry daily
        late_daily_new = daily_cum(df_a)
        days_replaced = days_late | days_filt
        base_remaining_daily = base_daily.copy()
        # Zero out replaced days
        for d in base_remaining_daily.index:
            ds = d.strftime('%Y%m%d')
            if ds in days_replaced:
                base_remaining_daily[d] = 0
        late_sys_daily = base_remaining_daily.add(late_daily_new.reindex(
            base_remaining_daily.index, fill_value=0))
        lines.append({"id":"late","label":f"Late Entry Rs.{int(late_sys_daily.sum()):,.0f}",
                      "color":colors[ci],"seriesType":"line",
                      "data":eq_pts(to_cum(late_sys_daily))})
        ci += 1

    if not df_b.empty:
        for tgt in [best_tgt_val]:
            sub = df_b[df_b['tgt_pct']==tgt]
            d = daily_cum(sub)
            lines.append({"id":f"tgt{int(tgt)}",
                          "label":f"Target {int(tgt)}% Rs.{int(sub['pnl'].sum()):,.0f}",
                          "color":colors[ci],"seriesType":"line",
                          "data":eq_pts(to_cum(d))})
            ci += 1

    if not df_c.empty:
        sub = df_c[df_c['variant']==best_var]
        d = daily_cum(sub)
        lines.append({"id":"partial","label":f"Partial {best_var} Rs.{int(sub['pnl'].sum()):,.0f}",
                      "color":colors[ci],"seriesType":"line",
                      "data":eq_pts(to_cum(d))})

    tv_json = {"isTvFormat":False,"candlestick":[],"volume":[],"lines":lines}
    send_custom_chart("120_improvements", tv_json, title="Improvement Tracks vs Base")

    # Save results
    if not df_a.empty: df_a.to_csv(f'{OUT_DIR}/120_late_entry.csv', index=False)
    if not df_b.empty: df_b.to_csv(f'{OUT_DIR}/120_target_sweep.csv', index=False)
    if not df_c.empty: df_c.to_csv(f'{OUT_DIR}/120_partial_book.csv', index=False)
    base_cpr.to_csv(f'{OUT_DIR}/120_cpr_sizing.csv', index=False)
    print(f"\n  Saved: 120_late_entry.csv | 120_target_sweep.csv | "
          f"120_partial_book.csv | 120_cpr_sizing.csv")
    print("\nDone.")


if __name__ == '__main__':
    main()
