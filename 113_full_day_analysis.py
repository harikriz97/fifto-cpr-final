"""
113_full_day_analysis.py — Full day-level analysis for ALL trading days
=======================================================================
Goal: Increase quality trades — find patterns across ALL days (not just traded)

For every day compute:
  CPR    : width, class, gap, trend, open position
  IB     : range, narrow/wide, failure signal
  Ichimoku: cloud position, TK cross, cloud color
  VP     : open vs VAH/VAL (value area)
  Intraday: first-hour range, OR (opening range), ADR
  Option MAE/MFE (per 15-min bucket from entry) — traded days only
  EOD state: where was price/option at 12:00, 13:00, 14:00, 14:30

Output: Excel (5 sheets)
  1. all_days         — every day, all features + trade outcome
  2. cpr_width_buckets — WR / avg P&L by CPR width decile
  3. eod_analysis      — for EOD-exit trades, what snapshot at 14:00 would have saved us
  4. mae_buckets       — MAE at +30min / +60min vs final outcome
  5. correlations      — feature vs win/loss matrix
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
VP_BINS    = 200
VP_VA_PCT  = 0.70
YEARS      = 5

def r2(v): return round(float(v), 2)
def get_atm(spot): return int(round(spot / STRIKE_INT) * STRIKE_INT)
def t2m(tstr):
    if not tstr: return None
    h, m, s = map(int, tstr.split(':'))
    return h * 60 + m + s / 60


def compute_vp(tks_prev):
    prices = tks_prev['price'].values
    if len(prices) < 10: return None, None, None
    p_min, p_max = prices.min(), prices.max()
    if p_max <= p_min: return None, None, None
    bins = np.linspace(p_min, p_max, VP_BINS + 1)
    bc   = (bins[:-1] + bins[1:]) / 2
    vols = tks_prev['volume'].values
    counts = np.zeros(VP_BINS)
    for i, p in enumerate(prices):
        idx = min(np.searchsorted(bins[1:], p), VP_BINS - 1)
        counts[idx] += max(vols[i], 1)
    poc_idx = int(np.argmax(counts))
    total = counts.sum(); target = total * VP_VA_PCT
    vah_i = poc_idx; val_i = poc_idx; cum = counts[poc_idx]
    while cum < target:
        u = vah_i + 1 < VP_BINS; d = val_i - 1 >= 0
        if u and d:
            if counts[vah_i+1] >= counts[val_i-1]: vah_i += 1; cum += counts[vah_i]
            else: val_i -= 1; cum += counts[val_i]
        elif u: vah_i += 1; cum += counts[vah_i]
        elif d: val_i -= 1; cum += counts[val_i]
        else: break
    return r2(bc[poc_idx]), r2(bc[vah_i]), r2(bc[val_i])


def _worker(args):
    date_str, meta, trade_row = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return None
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')].copy()
        if len(day) < 30: return None

        spot_o = r2(day.iloc[0]['price'])
        spot_h = r2(day['price'].max())
        spot_l = r2(day['price'].min())
        spot_c = r2(day.iloc[-1]['price'])
        day_range = r2(spot_h - spot_l)

        # IB (9:15-9:45)
        ib = day[day['time'] <= '09:45:00']
        ib_h = r2(ib['price'].max()) if not ib.empty else None
        ib_l = r2(ib['price'].min()) if not ib.empty else None
        ib_range = r2(ib_h - ib_l) if ib_h and ib_l else None

        # First-hour range (9:15-10:15)
        fhr = day[day['time'] <= '10:15:00']
        fh_range = r2(fhr['price'].max() - fhr['price'].min()) if not fhr.empty else None

        # OR (9:15-9:30)
        or_tks = day[day['time'] <= '09:30:00']
        or_range = r2(or_tks['price'].max() - or_tks['price'].min()) if not or_tks.empty else None

        # Open vs CPR
        tc  = meta.get('tc'); bc_ = meta.get('bc')
        pvt = meta.get('pvt')
        cpr_width = r2(tc - bc_) if tc and bc_ else None
        open_pos = None
        if tc and bc_:
            if   spot_o > tc:  open_pos = 'above_tc'
            elif spot_o < bc_: open_pos = 'below_bc'
            else:              open_pos = 'inside_cpr'

        # Open vs key R/S levels
        r1 = meta.get('r1'); s1 = meta.get('s1')
        r2_ = meta.get('r2'); s2 = meta.get('s2')

        # VP: open vs value area
        vah = meta.get('vah'); val = meta.get('val')
        vp_pos = None
        if vah and val:
            if   spot_o > vah: vp_pos = 'above_vah'
            elif spot_o < val: vp_pos = 'below_val'
            else:              vp_pos = 'inside_va'

        # Ichimoku
        ichi_sig = meta.get('ichi_sig')

        # CPR width class (filled later with median)
        cpr_class = meta.get('cpr_class')

        # Was there a trade?
        traded    = trade_row is not None
        opt       = trade_row['opt']         if traded else None
        entry_t   = trade_row['entry_time']  if traded else None
        exit_r    = trade_row['exit_reason'] if traded else None
        pnl       = trade_row['pnl']         if traded else None
        win       = bool(pnl > 0)            if traded else None
        ep        = trade_row['entry_price'] if traded else None
        xp        = trade_row['exit_price']  if traded else None
        zone      = trade_row['zone']        if traded else None

        # Option micro-level MAE/MFE snapshots (traded days)
        snap_keys = ['opt_15m','opt_30m','opt_45m','opt_60m','opt_90m',
                     'opt_120m','opt_at_1200','opt_at_1300','opt_at_1400','opt_at_1430']
        snaps = {k: None for k in snap_keys}
        opt_mae_30m = None; opt_mfe_30m = None
        opt_mae_60m = None; opt_mfe_60m = None
        spot_at_entry = None; spot_mae = None; spot_mfe = None
        exit_time = None; time_in_trade = None

        if traded and entry_t and ep and ep > 0:
            expiries = list_expiry_dates(date_str, index_name='NIFTY')
            if expiries:
                spot_at = day[day['time'] >= entry_t]
                if not spot_at.empty:
                    spot_at_entry = r2(spot_at.iloc[0]['price'])
                    atm   = get_atm(spot_at_entry)
                    instr = f'NIFTY{expiries[0]}{atm}{opt}'
                    opt_tks = load_tick_data(date_str, instr, entry_t)
                    if opt_tks is not None and not opt_tks.empty:
                        os_  = opt_tks[opt_tks['time'] >= entry_t].reset_index(drop=True)
                        if not os_.empty:
                            entry_m = t2m(entry_t)
                            ps = os_['price'].values
                            ts = os_['time'].values

                            # Snapshots at fixed offsets
                            for offset, key in [(15,'opt_15m'),(30,'opt_30m'),(45,'opt_45m'),
                                                (60,'opt_60m'),(90,'opt_90m'),(120,'opt_120m')]:
                                tgt_m = entry_m + offset
                                sub = [p for t_, p in zip(ts, ps) if t2m(t_) >= tgt_m]
                                snaps[key] = r2(sub[0] / ep) if sub else None

                            # Snapshots at clock times
                            for ct, key in [('12:00:00','opt_at_1200'),('13:00:00','opt_at_1300'),
                                            ('14:00:00','opt_at_1400'),('14:30:00','opt_at_1430')]:
                                sub = os_[os_['time'] >= ct]
                                snaps[key] = r2(sub.iloc[0]['price'] / ep) if not sub.empty else None

                            # MAE/MFE at 30min and 60min
                            def window_maemfe(minutes):
                                tgt_m = entry_m + minutes
                                sub = [p for t_, p in zip(ts, ps) if t2m(t_) <= tgt_m]
                                if not sub: return None, None
                                sub = np.array(sub)
                                mae = r2((sub.max() - ep) / ep * 100)   # % rise = adverse for seller
                                mfe = r2((ep - sub.min()) / ep * 100)   # % drop = favorable
                                return mae, mfe

                            opt_mae_30m, opt_mfe_30m = window_maemfe(30)
                            opt_mae_60m, opt_mfe_60m = window_maemfe(60)

                            # Exit time
                            EOD = '15:20:00'
                            for t_, p in zip(ts, ps):
                                if t_ >= EOD:
                                    exit_time = t_; break
                                if (ep - p) / ep >= 0.30:
                                    exit_time = t_; break
                                if (p - ep) / ep >= 1.00:
                                    exit_time = t_; break
                            if exit_time is None and len(ts): exit_time = ts[-1]
                            if exit_time:
                                time_in_trade = r2(t2m(exit_time) - t2m(entry_t))

                # Spot MAE/MFE
                if spot_at_entry:
                    after = day[day['time'] >= entry_t]['price'].values
                    if opt == 'CE':
                        spot_mfe = r2(spot_at_entry - after.min())
                        spot_mae = r2(after.max() - spot_at_entry)
                    else:
                        spot_mfe = r2(after.max() - spot_at_entry)
                        spot_mae = r2(spot_at_entry - after.min())

        row = {
            'date': date_str, 'year': date_str[:4],
            # Spot OHLC
            'spot_open': spot_o, 'spot_high': spot_h, 'spot_low': spot_l, 'spot_close': spot_c,
            'day_range': day_range, 'fh_range': fh_range, 'or_range': or_range,
            # IB
            'ib_h': ib_h, 'ib_l': ib_l, 'ib_range': ib_range,
            'ib_class': meta.get('ib_class'),
            # CPR
            'cpr_width': cpr_width, 'cpr_class': cpr_class, 'open_pos': open_pos,
            'tc': tc, 'bc': bc_, 'pvt': pvt,
            'r1': r1, 'r2': r2_, 's1': s1, 's2': s2,
            'cpr_gap': meta.get('cpr_gap'), 'cpr_trend': meta.get('cpr_trend'),
            'inside_cpr': meta.get('inside_cpr'),
            # VP
            'vah': vah, 'val': val, 'poc': meta.get('poc'), 'vp_pos': vp_pos,
            # Ichimoku
            'ichi_sig': ichi_sig,
            # Trade outcome
            'traded': traded, 'zone': zone, 'opt': opt,
            'entry_time': entry_t, 'exit_time': exit_time,
            'exit_reason': exit_r, 'pnl': pnl, 'win': win,
            'ep': ep, 'xp': xp,
            'time_in_trade_min': time_in_trade,
            'spot_at_entry': spot_at_entry, 'spot_mae': spot_mae, 'spot_mfe': spot_mfe,
            # Option snapshots
            **snaps,
            'opt_mae_30m_pct': opt_mae_30m, 'opt_mfe_30m_pct': opt_mfe_30m,
            'opt_mae_60m_pct': opt_mae_60m, 'opt_mfe_60m_pct': opt_mfe_60m,
        }
        return row
    except Exception:
        return None


def main():
    print("Phase 1: Building daily metadata (OHLC, CPR, Ichimoku, VP)...")
    t0 = time.time()
    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
    seed_idx  = max(0, all_dates.index(dates_5yr[0]) - 110)
    scan_all  = all_dates[seed_idx:]

    ohlc_rows = []; prev_tks = None
    for d in scan_all:
        tks = load_spot_data(d, 'NIFTY')
        if tks is None: continue
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')]
        if len(day) < 2: continue
        ohlc_rows.append({
            'date': d, 'o': day.iloc[0]['price'],
            'h': day['price'].max(), 'l': day['price'].min(), 'c': day.iloc[-1]['price'],
            'prev_tks': prev_tks,
        })
        prev_tks = day

    df_d = pd.DataFrame([{k: v for k, v in r.items() if k != 'prev_tks'} for r in ohlc_rows])
    prev_tks_list = [r['prev_tks'] for r in ohlc_rows]

    # CPR
    df_d['pdh'] = df_d['h'].shift(1); df_d['pdl'] = df_d['l'].shift(1); df_d['pdc'] = df_d['c'].shift(1)
    df_d['pvt'] = ((df_d['pdh']+df_d['pdl']+df_d['pdc'])/3).round(2)
    df_d['bc']  = ((df_d['pdh']+df_d['pdl'])/2).round(2)
    df_d['tc']  = (2*df_d['pvt']-df_d['bc']).round(2)
    df_d['r1']  = (2*df_d['pvt']-df_d['pdl']).round(2)
    df_d['r2']  = (df_d['pvt']+(df_d['pdh']-df_d['pdl'])).round(2)
    df_d['s1']  = (2*df_d['pvt']-df_d['pdh']).round(2)
    df_d['s2']  = (df_d['pvt']-(df_d['pdh']-df_d['pdl'])).round(2)
    df_d['cpr_gap']    = ((df_d['tc'].shift(1) < df_d['bc']) | (df_d['bc'].shift(1) > df_d['tc'])).astype(int)
    df_d['inside_cpr'] = ((df_d['tc'].shift(1) < df_d['tc'].shift(2)) & (df_d['bc'].shift(1) > df_d['bc'].shift(2))).astype(int)
    cpr_mid = (df_d['tc'] + df_d['bc']) / 2
    df_d['cpr_trend'] = ((cpr_mid.shift(1) > cpr_mid.shift(2)) & (cpr_mid.shift(2) > cpr_mid.shift(3))).map({True:'up',False:'down'})

    # Ichimoku
    def rolling_mid(h, l, n):
        return ((h.rolling(n).max() + l.rolling(n).min()) / 2).round(2)
    df_d['tenkan'] = rolling_mid(df_d['h'], df_d['l'], 9)
    df_d['kijun']  = rolling_mid(df_d['h'], df_d['l'], 26)
    df_d['span_a'] = ((df_d['tenkan'].shift(26) + df_d['kijun'].shift(26)) / 2).round(2)
    df_d['span_b'] = rolling_mid(df_d['h'].shift(26), df_d['l'].shift(26), 52)
    df_d['cloud_top']    = df_d[['span_a','span_b']].max(axis=1)
    df_d['cloud_bottom'] = df_d[['span_a','span_b']].min(axis=1)
    def cloud_sig(row):
        if pd.isna(row['cloud_top']): return None
        if row['o'] > row['cloud_top']:    return 'PE'
        if row['o'] < row['cloud_bottom']: return 'CE'
        return 'inside'
    df_d['ichi_sig'] = df_d.apply(cloud_sig, axis=1)

    # IB range for median (load separately for speed: just from OHLC rows)
    # We'll compute IB range in the worker; use ib_range from day ticks
    # For now skip IB median (compute after all workers)

    # Volume Profile
    print(f"  Computing VP for {len(df_d)} days...")
    poc_l=[]; vah_l=[]; val_l=[]
    for pt in prev_tks_list:
        if pt is not None:
            poc, vah, val = compute_vp(pt)
        else:
            poc = vah = val = None
        poc_l.append(poc); vah_l.append(vah); val_l.append(val)
    df_d['poc'] = poc_l; df_d['vah'] = vah_l; df_d['val'] = val_l

    # CPR width median for class
    cpr_widths = (df_d['tc'] - df_d['bc']).abs()
    cpr_med = cpr_widths[df_d['date'].isin(dates_5yr)].median()
    df_d['cpr_class'] = cpr_widths.apply(lambda x: 'narrow' if pd.notna(x) and x < cpr_med else 'wide')

    # IB range median (approximate: use day_range / 3 proxy; exact computed in worker)
    ib_med_approx = 82.7  # from script 109

    meta_map = {}
    for _, row in df_d.iterrows():
        d = row['date']
        meta_map[d] = {
            'pvt': row.get('pvt'), 'tc': row.get('tc'), 'bc': row.get('bc'),
            'r1': row.get('r1'), 'r2': row.get('r2'), 's1': row.get('s1'), 's2': row.get('s2'),
            'cpr_gap': row.get('cpr_gap'), 'cpr_trend': row.get('cpr_trend'),
            'inside_cpr': row.get('inside_cpr'), 'cpr_class': row.get('cpr_class'),
            'ichi_sig': row.get('ichi_sig'),
            'poc': row.get('poc'), 'vah': row.get('vah'), 'val': row.get('val'),
            'ib_class': None,  # filled below per day
        }
    print(f"  Metadata ready in {time.time()-t0:.0f}s | CPR median: {cpr_med:.1f} pts")

    # Load base trades
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date_str'] = pd.to_datetime(base['date'].astype(str), format='mixed').dt.strftime('%Y%m%d')
    base = base.rename(columns={'pnl_conv': 'pnl'})
    trade_map = {}
    for _, row in base.iterrows():
        trade_map[row['date_str']] = {
            'opt': row['opt'], 'zone': row['zone'],
            'entry_time': row['entry_time'], 'exit_reason': row['exit_reason'],
            'entry_price': row['entry_price'], 'exit_price': row['exit_price'],
            'pnl': row['pnl'],
        }

    # Build args
    args_list = [(d, meta_map.get(d, {}), trade_map.get(d)) for d in dates_5yr]

    print(f"\nPhase 2: Enriching {len(args_list)} days (parallel)...")
    t0 = time.time()
    n_workers = min(16, cpu_count() or 4)
    with Pool(processes=n_workers) as pool:
        results = pool.map(_worker, args_list)
    print(f"  Done in {time.time()-t0:.0f}s")

    df = pd.DataFrame([r for r in results if r is not None])
    if df.empty:
        print("No data."); return

    # IB class using actual ib_range median
    ib_med = df['ib_range'].median()
    df['ib_class'] = df['ib_range'].apply(lambda x: 'narrow' if pd.notna(x) and x < ib_med else 'wide')
    print(f"  IB range median: {ib_med:.1f} pts")

    # ── Print key stats ────────────────────────────────────────────────────────
    traded = df[df['traded']==True]
    blank  = df[df['traded']==False]
    print(f"\n  Total days: {len(df)} | Traded: {len(traded)} | Blank: {len(blank)}")
    print(f"  Overall WR: {traded['win'].mean()*100:.1f}% | Total P&L: Rs.{traded['pnl'].sum():,.0f}")

    # CPR width decile analysis
    print(f"\n  CPR width decile vs WR (traded days):")
    traded2 = traded.dropna(subset=['cpr_width'])
    traded2 = traded2.copy()
    traded2['cpr_decile'] = pd.qcut(traded2['cpr_width'], 5, labels=['D1(narrow)','D2','D3','D4','D5(wide)'])
    for dec, g in traded2.groupby('cpr_decile'):
        print(f"    {dec}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Avg Rs.{g['pnl'].mean():.0f} | CPR {g['cpr_width'].mean():.0f}pts")

    # EOD analysis: what was opt price at 14:00 for EOD-exit trades?
    eod_trades = traded[traded['exit_reason']=='eod'].copy()
    eod_trades = eod_trades.dropna(subset=['opt_at_1400'])
    print(f"\n  EOD exit analysis ({len(eod_trades)} trades with 14:00 snapshot):")
    # If opt at 14:00 is > 1.0*entry → likely a loser
    eod_trades['state_1400'] = eod_trades['opt_at_1400'].apply(
        lambda x: 'winning' if x < 1.0 else 'losing')
    for st, g in eod_trades.groupby('state_1400'):
        print(f"    At 14:00 {st}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Avg Rs.{g['pnl'].mean():.0f}")
    # Simulated: exit at 14:00 if losing
    eod_sim = eod_trades.copy()
    # If at 14:00 opt > entry (losing), exit at 14:00 price vs current exit
    losing_eod = eod_trades[eod_trades['opt_at_1400'] > 1.0]
    if len(losing_eod):
        # At 14:00 they're showing opt_at_1400 ratio. Current xp known.
        # Simulated exit at 14:00: pnl = (ep - ep*opt_at_1400)*LOT_SIZE*SCALE
        sim_pnl = ((losing_eod['ep'] - losing_eod['ep']*losing_eod['opt_at_1400']) * LOT_SIZE * SCALE).sum()
        actual_pnl = losing_eod['pnl'].sum()
        print(f"\n    If we exit losing EOD trades at 14:00:")
        print(f"      Actual P&L:    Rs.{actual_pnl:,.0f}")
        print(f"      Simulated exit: Rs.{sim_pnl:,.0f}")
        print(f"      Improvement:    Rs.{sim_pnl - actual_pnl:,.0f}")

    # MAE at 30min as predictor
    print(f"\n  opt_mae_30m_pct as early exit signal:")
    t30 = traded.dropna(subset=['opt_mae_30m_pct'])
    for thr in [25, 50, 75, 100]:
        sub_keep = t30[t30['opt_mae_30m_pct'] <= thr]
        sub_exit = t30[t30['opt_mae_30m_pct'] > thr]
        if len(sub_keep) == 0: continue
        improvement = sub_keep['pnl'].sum() - t30['pnl'].sum()  # vs keeping all
        print(f"    Exit if 30m MAE > {thr}%: keep {len(sub_keep)}t WR {sub_keep['win'].mean()*100:.1f}% "
              f"Avg Rs.{sub_keep['pnl'].mean():.0f} | skip {len(sub_exit)}t")

    # Blank day feature distribution vs base day
    print(f"\n  Feature comparison: Base days vs Blank days")
    for feat in ['cpr_class', 'open_pos', 'ichi_sig', 'vp_pos', 'ib_class']:
        print(f"\n    {feat}:")
        base_dist  = traded[feat].value_counts(normalize=True).round(3) if feat in traded else {}
        blank_dist = blank[feat].value_counts(normalize=True).round(3)  if feat in blank  else {}
        all_vals   = set(list(base_dist.index) + list(blank_dist.index))
        for v in sorted(all_vals, key=str):
            bv = base_dist.get(v, 0); blv = blank_dist.get(v, 0)
            print(f"      {str(v):<20}: base {bv*100:5.1f}%  blank {blv*100:5.1f}%")

    # ── Build summary sheets ───────────────────────────────────────────────────
    def grp(g):
        t = g[g['traded']==True]
        return pd.Series({
            'total_days': len(g), 'traded_days': len(t), 'blank_days': len(g)-len(t),
            'win_rate':   round(t['win'].mean()*100, 1) if len(t) else None,
            'total_pnl':  round(t['pnl'].sum(), 0)     if len(t) else None,
            'avg_pnl':    round(t['pnl'].mean(), 0)    if len(t) else None,
            'opt_mae_30m_avg': round(t['opt_mae_30m_pct'].mean(), 1) if len(t) else None,
            'opt_mae_60m_avg': round(t['opt_mae_60m_pct'].mean(), 1) if len(t) else None,
        })

    df['cpr_decile'] = pd.qcut(df['cpr_width'].fillna(df['cpr_width'].median()),
                               5, labels=['D1','D2','D3','D4','D5'], duplicates='drop')

    by_cpr     = df.groupby('cpr_decile').apply(grp).reset_index()
    by_open    = df.groupby('open_pos').apply(grp).reset_index()
    by_ichi    = df.groupby('ichi_sig').apply(grp).reset_index()
    by_vp      = df.groupby('vp_pos').apply(grp).reset_index()
    by_year    = df.groupby('year').apply(grp).reset_index()
    by_ib      = df.groupby('ib_class').apply(grp).reset_index()

    # EOD analysis sheet
    eod_full = traded[traded['exit_reason']=='eod'][
        ['date','opt','zone','entry_time','exit_time','ep','xp','pnl','win',
         'opt_15m','opt_30m','opt_45m','opt_60m','opt_90m','opt_120m',
         'opt_at_1200','opt_at_1300','opt_at_1400','opt_at_1430',
         'opt_mae_30m_pct','opt_mae_60m_pct','cpr_width','cpr_class','ib_range']
    ].copy()

    # Hard SL analysis sheet
    sl_full = traded[traded['exit_reason']=='hard_sl'][
        ['date','opt','zone','entry_time','exit_time','ep','xp','pnl',
         'opt_15m','opt_30m','opt_45m','opt_60m',
         'opt_mae_30m_pct','opt_mae_60m_pct','cpr_width','spot_mae','ib_range']
    ].copy()

    # Correlation matrix (numeric features vs win)
    num_cols = ['cpr_width','ib_range','fh_range','or_range','day_range',
                'opt_mae_30m_pct','opt_mae_60m_pct','spot_mae','spot_mfe','time_in_trade_min']
    corr_df = traded[['win']+[c for c in num_cols if c in traded.columns]].copy()
    corr_df['win'] = corr_df['win'].astype(float)
    corr_matrix = corr_df.corr()[['win']].round(3).sort_values('win', ascending=False)

    # ── Export to Excel ────────────────────────────────────────────────────────
    col_order = [
        'date','year','traded','zone','opt','open_pos','cpr_class','ib_class','ichi_sig','vp_pos',
        'spot_open','spot_high','spot_low','spot_close','day_range','fh_range','or_range',
        'ib_h','ib_l','ib_range',
        'cpr_width','tc','bc','pvt','r1','r2','s1','s2',
        'vah','val','poc',
        'entry_time','exit_time','exit_reason','ep','xp','pnl','win','time_in_trade_min',
        'spot_at_entry','spot_mae','spot_mfe',
        'opt_15m','opt_30m','opt_45m','opt_60m','opt_90m','opt_120m',
        'opt_at_1200','opt_at_1300','opt_at_1400','opt_at_1430',
        'opt_mae_30m_pct','opt_mfe_30m_pct','opt_mae_60m_pct','opt_mfe_60m_pct',
        'cpr_gap','cpr_trend','inside_cpr',
    ]
    col_order = [c for c in col_order if c in df.columns]

    excel_path = f'{OUT_DIR}/113_full_day_analysis.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df[col_order].sort_values('date').to_excel(writer, sheet_name='all_days',    index=False)
        by_cpr.to_excel(writer,    sheet_name='by_cpr_decile', index=False)
        by_open.to_excel(writer,   sheet_name='by_open_pos',   index=False)
        by_ichi.to_excel(writer,   sheet_name='by_ichimoku',   index=False)
        by_vp.to_excel(writer,     sheet_name='by_vp_pos',     index=False)
        by_ib.to_excel(writer,     sheet_name='by_ib_class',   index=False)
        by_year.to_excel(writer,   sheet_name='by_year',       index=False)
        eod_full.to_excel(writer,  sheet_name='eod_exits',     index=False)
        sl_full.to_excel(writer,   sheet_name='hard_sl_exits', index=False)
        corr_matrix.to_excel(writer, sheet_name='correlations')

    print(f"\n  Saved → {excel_path}")
    print(f"  Sheets: all_days | by_cpr_decile | by_open_pos | by_ichimoku | by_vp_pos |")
    print(f"          by_ib_class | by_year | eod_exits | hard_sl_exits | correlations")
    print("\nDone.")

if __name__ == '__main__':
    main()
