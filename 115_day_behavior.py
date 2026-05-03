"""
115_day_behavior.py — Day-by-day behavior analysis + pattern observations
=========================================================================
For every trading day:
  1. Classify the day type (trend / range / reversal / gap / CPR-magnet)
  2. Write an automated observation string
  3. Measure price interaction with CPR / R1 / S1 / IB
  4. Record option outcome (if traded)

Then:
  - Group days by behavior type → WR / avg P&L per type
  - Find blank days that LOOK LIKE the profitable base-day patterns
  - Find patterns currently missed (high-quality blank day profile)
  - Export full Excel for manual study
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from my_util import load_spot_data, list_trading_dates

OUT_DIR = 'data/20260430'


# ── Worker: one day ────────────────────────────────────────────────────────────
def _worker(args):
    date_str, meta, trade_row = args
    os.environ.pop('MESSAGE_CALLBACK_URL', None)
    try:
        tks = load_spot_data(date_str, 'NIFTY')
        if tks is None: return None
        day = tks[(tks['time'] >= '09:15:00') & (tks['time'] <= '15:30:00')].copy()
        if len(day) < 30: return None

        px = day['price'].values
        ts = day['time'].values
        spot_o = round(float(px[0]),  2)
        spot_h = round(float(px.max()),2)
        spot_l = round(float(px.min()),2)
        spot_c = round(float(px[-1]), 2)
        day_range = round(spot_h - spot_l, 2)

        def price_at(t_str):
            sub = day[day['time'] >= t_str]
            return round(float(sub.iloc[0]['price']), 2) if not sub.empty else None

        p_1000 = price_at('10:00:00')
        p_1100 = price_at('11:00:00')
        p_1200 = price_at('12:00:00')
        p_1300 = price_at('13:00:00')
        p_1400 = price_at('14:00:00')
        p_1430 = price_at('14:30:00')

        # IB
        ib = day[day['time'] <= '09:45:00']
        ib_h = round(float(ib['price'].max()), 2) if not ib.empty else spot_o
        ib_l = round(float(ib['price'].min()), 2) if not ib.empty else spot_o
        ib_range = round(ib_h - ib_l, 2)

        # IB expansion: did post-9:45 price exceed IB?
        post_ib = day[day['time'] > '09:45:00']
        ib_exp_up   = bool(not post_ib.empty and post_ib['price'].max() > ib_h)
        ib_exp_down = bool(not post_ib.empty and post_ib['price'].min() < ib_l)
        ib_expand = 'both' if (ib_exp_up and ib_exp_down) else \
                    'up' if ib_exp_up else 'down' if ib_exp_down else 'none'

        # First-hour direction (9:15 → 10:15)
        p_fh_end = price_at('10:15:00') or spot_c
        fh_chg   = round((p_fh_end - spot_o) / spot_o * 100, 2)
        fh_dir   = 'up' if fh_chg > 0.15 else 'down' if fh_chg < -0.15 else 'flat'

        # Day direction
        day_chg  = round((spot_c - spot_o) / spot_o * 100, 2)
        day_dir  = 'up' if day_chg > 0.30 else 'down' if day_chg < -0.30 else 'flat'

        # CPR levels
        tc  = meta.get('tc');  bc  = meta.get('bc'); pvt = meta.get('pvt')
        r1  = meta.get('r1');  s1  = meta.get('s1')
        r2  = meta.get('r2');  s2  = meta.get('s2')

        # Open position
        if tc and bc:
            if   spot_o > tc:  open_pos = 'above_tc'
            elif spot_o < bc:  open_pos = 'below_bc'
            else:              open_pos = 'inside_cpr'
        else:
            open_pos = 'unknown'

        cpr_width = round(float(tc - bc), 2) if tc and bc else None

        # CPR interaction: did price TOUCH CPR zone during day?
        cpr_touch = False
        if tc and bc:
            sub = day[(day['price'] >= bc) & (day['price'] <= tc)]
            cpr_touch = not sub.empty

        # CPR retest: opened outside, came back to CPR?
        cpr_retest = False
        if open_pos != 'inside_cpr' and cpr_touch:
            cpr_retest = True

        # R1 / S1 touch
        tol = 0.002
        def level_touch(lvl):
            if not lvl: return False
            return any(abs(p - lvl) / p <= tol for p in px)

        r1_touch = level_touch(r1)
        s1_touch = level_touch(s1)
        r2_touch = level_touch(r2)
        s2_touch = level_touch(s2)

        # Time of day high / low
        def time_of_extreme(kind):
            idx = int(np.argmax(px)) if kind == 'high' else int(np.argmin(px))
            return ts[idx] if idx < len(ts) else None

        high_time = time_of_extreme('high')
        low_time  = time_of_extreme('low')

        # Day type classification
        # Trend up: open above CPR, high after 12:00, close near high
        # Trend down: open below CPR, low after 12:00, close near low
        # Range: IB holds, price stays within 0.5% of open all day
        # V-reversal up: first half down, second half up
        # V-reversal down: first half up, second half down
        # Gap-and-go: large gap, continues in gap direction
        # CPR-magnet: price oscillates around CPR midpoint

        range_pct = day_range / spot_o * 100
        high_in_pm = high_time and high_time >= '12:00:00'
        low_in_pm  = low_time  and low_time  >= '12:00:00'

        # V-reversal detection: first-half direction opposite to close
        fh_up   = fh_chg >  0.20
        fh_down = fh_chg < -0.20
        v_rev_up   = bool(fh_down and day_dir == 'up')
        v_rev_down = bool(fh_up   and day_dir == 'down')

        # Range day: close within 0.3% of open AND range < 0.8%
        range_day = bool(abs(day_chg) < 0.30 and range_pct < 0.80)

        # Gap from prev close
        pdh = meta.get('pdh'); pdl = meta.get('pdl'); pdc = meta.get('pdc')
        gap_pct = None
        if pdc:
            gap_pct = round((spot_o - pdc) / pdc * 100, 2)
        gap_type = None
        if gap_pct is not None:
            if   gap_pct >  0.30: gap_type = 'gap_up'
            elif gap_pct < -0.30: gap_type = 'gap_down'
            else:                  gap_type = 'flat_open'

        # Classify day type
        if v_rev_up:
            day_type = 'V_reversal_up'
        elif v_rev_down:
            day_type = 'V_reversal_down'
        elif range_day:
            day_type = 'range_day'
        elif day_dir == 'up' and open_pos in ('above_tc', 'inside_cpr') and not low_in_pm:
            day_type = 'trend_up'
        elif day_dir == 'down' and open_pos in ('below_bc', 'inside_cpr') and not high_in_pm:
            day_type = 'trend_down'
        elif cpr_retest and abs(day_chg) < 0.50:
            day_type = 'CPR_magnet'
        elif day_dir == 'up':
            day_type = 'normal_up'
        elif day_dir == 'down':
            day_type = 'normal_down'
        else:
            day_type = 'flat'

        # ── Write observation ──────────────────────────────────────────────────
        obs_parts = []
        obs_parts.append(f"Open {open_pos.replace('_',' ')}")
        if gap_type and gap_type != 'flat_open':
            obs_parts.append(f"{gap_type.replace('_',' ')} ({gap_pct:+.2f}%)")
        obs_parts.append(f"First-hr {fh_dir} ({fh_chg:+.2f}%)")
        obs_parts.append(f"IB {ib_expand}")
        if cpr_retest: obs_parts.append("CPR retest")
        if r1_touch:   obs_parts.append("R1 touched")
        if s1_touch:   obs_parts.append("S1 touched")
        if r2_touch:   obs_parts.append("R2 touched")
        if s2_touch:   obs_parts.append("S2 touched")
        obs_parts.append(f"Day {day_type} close {day_chg:+.2f}%")
        observation = " → ".join(obs_parts)

        # Trade outcome
        traded    = trade_row is not None
        opt       = trade_row['opt']          if traded else None
        entry_t   = trade_row['entry_time']   if traded else None
        exit_r    = trade_row['exit_reason']  if traded else None
        pnl       = trade_row['pnl']          if traded else None
        win       = bool(pnl > 0)             if traded else None
        zone      = trade_row['zone']         if traded else None

        if traded:
            outcome = f"{opt} {zone} → {exit_r} Rs.{int(pnl):+,}"
            observation += f"  ||  TRADE: {outcome}"

        return {
            'date': date_str, 'year': date_str[:4],
            'spot_open': spot_o, 'spot_high': spot_h, 'spot_low': spot_l, 'spot_close': spot_c,
            'day_range': day_range, 'range_pct': round(range_pct, 2),
            'day_chg_pct': day_chg, 'fh_chg_pct': fh_chg,
            'day_dir': day_dir, 'fh_dir': fh_dir, 'day_type': day_type,
            'open_pos': open_pos, 'cpr_width': cpr_width,
            'ib_h': ib_h, 'ib_l': ib_l, 'ib_range': ib_range, 'ib_expand': ib_expand,
            'ib_exp_up': ib_exp_up, 'ib_exp_down': ib_exp_down,
            'cpr_touch': cpr_touch, 'cpr_retest': cpr_retest,
            'r1_touch': r1_touch, 'r2_touch': r2_touch,
            's1_touch': s1_touch, 's2_touch': s2_touch,
            'high_time': high_time, 'low_time': low_time,
            'gap_pct': gap_pct, 'gap_type': gap_type,
            'p_1000': p_1000, 'p_1100': p_1100, 'p_1200': p_1200,
            'p_1300': p_1300, 'p_1400': p_1400, 'p_1430': p_1430,
            'tc': tc, 'bc': bc, 'pvt': pvt, 'r1': r1, 'r2': r2, 's1': s1, 's2': s2,
            'ichi_sig': meta.get('ichi_sig'),
            'vp_pos': meta.get('vp_pos'),
            'v_rev_up': v_rev_up, 'v_rev_down': v_rev_down, 'range_day': range_day,
            'traded': traded, 'zone': zone, 'opt': opt,
            'entry_time': entry_t, 'exit_reason': exit_r, 'pnl': pnl, 'win': win,
            'observation': observation,
        }
    except Exception:
        return None


def main():
    print("Loading metadata from 113...")
    df_meta = pd.read_excel(f'{OUT_DIR}/113_full_day_analysis.xlsx',
                            sheet_name='all_days', dtype={'date': str})
    df_meta['date'] = df_meta['date'].astype(str)

    # Build meta map
    meta_map = {}
    for _, row in df_meta.iterrows():
        meta_map[row['date']] = {k: (None if pd.isna(v) else v)
                                 for k, v in row.items()
                                 if k in ('tc','bc','pvt','r1','r2','s1','s2',
                                          'pdh','pdl','pdc','ichi_sig','vp_pos',
                                          'vah','val','cpr_class','ib_class')}

    # Base trades
    base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
    base['date_str'] = pd.to_datetime(base['date'].astype(str), format='mixed').dt.strftime('%Y%m%d')
    base = base.rename(columns={'pnl_conv': 'pnl'})
    trade_map = {row['date_str']: row for _, row in base.iterrows()}

    all_dates = list_trading_dates()
    latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
    dates_5yr = [d for d in all_dates
                 if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=5)]

    args_list = [(d, meta_map.get(d, {}), trade_map.get(d)) for d in dates_5yr]
    print(f"Running behavior analysis on {len(args_list)} days...")
    t0 = time.time()
    with Pool(processes=min(16, cpu_count() or 4)) as pool:
        results = pool.map(_worker, args_list)
    print(f"  Done in {time.time()-t0:.0f}s")

    df = pd.DataFrame([r for r in results if r is not None])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  {len(df)} days classified")

    traded = df[df['traded'] == True]
    blank  = df[df['traded'] == False]

    # ── Stats by day type ─────────────────────────────────────────────────────
    def stats(sub, label=''):
        t = sub[sub['traded'] == True] if 'traded' in sub.columns else sub
        total = len(sub)
        tr = len(t)
        wr  = t['win'].mean()*100 if tr else 0
        tp  = t['pnl'].sum()      if tr else 0
        ap  = t['pnl'].mean()     if tr else 0
        return total, tr, round(wr,1), round(tp,0), round(ap,0)

    print(f"\n{'='*90}")
    print(f"  DAY TYPE ANALYSIS — {len(df)} total days | {len(traded)} traded | {len(blank)} blank")
    print(f"{'='*90}")
    print(f"  {'Day Type':<22} | {'Total':>5} | {'Traded':>6} | {'Blank':>5} | "
          f"{'WR':>6} | {'Total P&L':>10} | {'Avg':>7} | {'Trade%':>7}")
    print(f"  {'-'*88}")
    for dt, g in df.groupby('day_type'):
        tot, tr, wr, tp, ap = stats(g)
        bl = tot - tr
        tr_pct = round(tr/tot*100, 0)
        print(f"  {dt:<22} | {tot:>5} | {tr:>6} | {bl:>5} | "
              f"{wr:>5.1f}% | Rs.{tp:>8,.0f} | Rs.{ap:>5,.0f} | {tr_pct:>5.0f}%")

    # ── Best pattern: what makes a day highly tradeable? ─────────────────────
    print(f"\n  Open position vs WR:")
    for op, g in traded.groupby('open_pos'):
        print(f"    {op:<18}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Avg Rs.{g['pnl'].mean():.0f}")

    print(f"\n  IB expansion vs WR (traded days):")
    for ie, g in traded.groupby('ib_expand'):
        print(f"    {ie:<10}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Avg Rs.{g['pnl'].mean():.0f}")

    print(f"\n  Day type + open pos (traded, top combos):")
    combo = traded.groupby(['day_type','open_pos']).agg(
        trades=('pnl','count'), wr=('win','mean'), avg_pnl=('pnl','mean')).reset_index()
    combo['wr'] = (combo['wr']*100).round(1)
    combo['avg_pnl'] = combo['avg_pnl'].round(0)
    combo = combo.sort_values('wr', ascending=False)
    for _, r in combo[combo['trades']>=10].head(12).iterrows():
        print(f"    {r['day_type']:<22} + {r['open_pos']:<15}: "
              f"{int(r['trades'])}t | WR {r['wr']:.1f}% | Avg Rs.{int(r['avg_pnl'])}")

    # ── Blank day profile: which blank day types look like profitable traded days? ──
    print(f"\n  Blank day type distribution vs traded day type:")
    for dt in df['day_type'].unique():
        b_cnt = len(blank[blank['day_type']==dt])
        t_cnt = len(traded[traded['day_type']==dt])
        b_pct = round(b_cnt/len(blank)*100, 0)
        t_pct = round(t_cnt/len(traded)*100, 0)
        t_wr  = traded[traded['day_type']==dt]['win'].mean()*100 if t_cnt else 0
        print(f"    {dt:<22}: blank {b_cnt:>3} ({b_pct:>3.0f}%) | traded {t_cnt:>3} ({t_pct:>3.0f}%) | "
              f"traded WR {t_wr:.1f}%")

    # ── High-value blank days: blank days with profitable-looking profile ──────
    # Profile: same day_type as high-WR traded days + open outside CPR + IB expanding
    high_wr_types = ['trend_up', 'trend_down', 'normal_up', 'normal_down']
    target_blank = blank[
        (blank['day_type'].isin(high_wr_types)) &
        (blank['open_pos'].isin(['above_tc','below_bc'])) &
        (blank['ib_expand'].isin(['up','down']))
    ]
    print(f"\n  Blank days matching HIGH-WR profile (trend/normal + open outside CPR + IB expanding):")
    print(f"    {len(target_blank)} days found")
    if not target_blank.empty:
        for dt, g in target_blank.groupby('day_type'):
            print(f"    {dt}: {len(g)} blank days")
        print(f"\n  Sample observations (first 10):")
        for _, r in target_blank.head(10).iterrows():
            print(f"    {r['date']}: {r['observation'][:120]}")

    # ── Gap day analysis ───────────────────────────────────────────────────────
    print(f"\n  Gap type vs WR (traded days):")
    for gt, g in traded.dropna(subset=['gap_type']).groupby('gap_type'):
        print(f"    {gt:<12}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Avg Rs.{g['pnl'].mean():.0f}")

    # ── CPR retest vs WR ──────────────────────────────────────────────────────
    print(f"\n  CPR retest vs WR:")
    for cr, g in traded.groupby('cpr_retest'):
        label = 'CPR retest=Yes' if cr else 'CPR retest=No'
        print(f"    {label}: {len(g)}t | WR {g['win'].mean()*100:.1f}% | "
              f"Avg Rs.{g['pnl'].mean():.0f}")

    # ── V-reversal pattern ────────────────────────────────────────────────────
    v_rev = traded[(traded['v_rev_up']) | (traded['v_rev_down'])]
    print(f"\n  V-reversal days in traded set: {len(v_rev)} | "
          f"WR {v_rev['win'].mean()*100:.1f}% | Avg Rs.{v_rev['pnl'].mean():.0f}")

    # ── Worst pattern (losing trades concentration) ───────────────────────────
    losers = traded[traded['win']==False]
    print(f"\n  Losing trade concentration by day type:")
    for dt, g in losers.groupby('day_type'):
        print(f"    {dt:<22}: {len(g)} losses | total loss Rs.{g['pnl'].sum():,.0f} | "
              f"avg Rs.{g['pnl'].mean():.0f}")

    # ── Export Excel ──────────────────────────────────────────────────────────
    col_order = [
        'date','year','traded','day_type','open_pos','gap_type',
        'fh_dir','day_dir','ib_expand','cpr_retest','v_rev_up','v_rev_down','range_day',
        'spot_open','spot_high','spot_low','spot_close',
        'day_chg_pct','fh_chg_pct','day_range','range_pct',
        'ib_h','ib_l','ib_range','ib_exp_up','ib_exp_down',
        'cpr_width','tc','bc','pvt','r1','r2','s1','s2',
        'r1_touch','r2_touch','s1_touch','s2_touch','cpr_touch',
        'high_time','low_time','gap_pct',
        'p_1000','p_1100','p_1200','p_1300','p_1400','p_1430',
        'ichi_sig','vp_pos',
        'zone','opt','entry_time','exit_reason','pnl','win',
        'observation',
    ]
    col_order = [c for c in col_order if c in df.columns]

    # Summary tables
    def grp_stats(g):
        t = g[g['traded']==True]
        return pd.Series({
            'total_days': len(g), 'traded': len(t), 'blank': len(g)-len(t),
            'trade_rate_pct': round(len(t)/len(g)*100,1),
            'wr_pct':   round(t['win'].mean()*100,1) if len(t) else None,
            'total_pnl':round(t['pnl'].sum(),0)      if len(t) else None,
            'avg_pnl':  round(t['pnl'].mean(),0)     if len(t) else None,
            'losses':   int((t['win']==False).sum())  if len(t) else 0,
        })

    by_type    = df.groupby('day_type').apply(grp_stats).reset_index()
    by_open    = df.groupby('open_pos').apply(grp_stats).reset_index()
    by_ib_exp  = df.groupby('ib_expand').apply(grp_stats).reset_index()
    by_gap     = df.dropna(subset=['gap_type']).groupby('gap_type').apply(grp_stats).reset_index()
    by_year    = df.groupby('year').apply(grp_stats).reset_index()

    # Blank day candidate sheet
    target_blank_out = target_blank[col_order].copy() if not target_blank.empty else pd.DataFrame()

    excel_path = f'{OUT_DIR}/115_day_behavior.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df[col_order].to_excel(writer,        sheet_name='all_days_observations', index=False)
        by_type.to_excel(writer,              sheet_name='by_day_type',           index=False)
        by_open.to_excel(writer,              sheet_name='by_open_pos',           index=False)
        by_ib_exp.to_excel(writer,            sheet_name='by_ib_expand',          index=False)
        by_gap.to_excel(writer,               sheet_name='by_gap_type',           index=False)
        by_year.to_excel(writer,              sheet_name='by_year',               index=False)
        if not target_blank_out.empty:
            target_blank_out.to_excel(writer, sheet_name='blank_quality_days',    index=False)

    print(f"\n  Saved → {excel_path}")
    print(f"  Sheets: all_days_observations | by_day_type | by_open_pos |")
    print(f"          by_ib_expand | by_gap_type | by_year | blank_quality_days")
    print("\nDone.")


if __name__ == '__main__':
    main()
