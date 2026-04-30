"""
62_full_report.py
Complete strategy analysis:
  1. All strategies with LOT_SIZE=65
  2. Conviction sizing: when v17a + cam same day same direction → 2× lots
                        when v17a + cam + v2 all same direction  → 3× lots
  3. CPR + Camarilla combined concept
  4. Individual + combined equity curves
"""
import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
from plot_util import send_custom_chart

LOT = 65   # new NIFTY lot size

def r2(v): return round(float(v), 2)

# ─────────────────────────────────────────────
# Load all backtest data
# ─────────────────────────────────────────────
df_all  = pd.read_csv('data/56_combined_trades.csv', parse_dates=['date'])
df_iv2  = pd.read_csv('data/20260428/52_more_trades_backtest.csv')
df_cam54= pd.read_csv('data/20260428/54_camarilla_standalone.csv')
df_cam55= pd.read_csv('data/20260428/55_camarilla_touch.csv')

v17a = df_all[df_all.strategy=='v17a'].copy().sort_values('date').reset_index(drop=True)
cam  = df_all[df_all.strategy.isin(['cam_l3','cam_h3'])].copy()
iv2  = df_iv2[df_iv2.idea=='C_second_trade'].copy()

# Rescale from 75→65: pnl = (ep-xp)*lots → multiply by 65/75
SCALE = 65 / 75
v17a['pnl65'] = v17a['pnl'] * SCALE
cam['pnl65']  = cam['pnl']  * SCALE
iv2['pnl65']  = iv2['pnl']  * SCALE

# ─────────────────────────────────────────────
# SECTION 1: Individual strategy results (65 lot)
# ─────────────────────────────────────────────
print("=" * 65)
print("  INDIVIDUAL STRATEGY RESULTS (LOT=65)")
print("=" * 65)

def show(label, df, pnl_col='pnl65'):
    n   = len(df)
    wr  = r2((df[pnl_col] > 0).mean() * 100)
    pnl = r2(df[pnl_col].sum())
    avg = r2(df[pnl_col].mean())
    eq  = df.sort_values('date')[pnl_col].cumsum()
    mdd = r2((eq - eq.cummax()).min())
    print(f"  {label:<28} {n:>4}t | WR {wr:>5.1f}% | P&L Rs {pnl:>10,.0f} | Avg {avg:>7,.0f} | DD {mdd:>10,.0f}")

show("v17a (CPR+EMA Sell)",       v17a)
show("cam_l3 (L3 touch→PE)",      cam[cam.strategy=='cam_l3'])
show("cam_h3 (H3 touch→CE)",      cam[cam.strategy=='cam_h3'])
show("intraday_v2 (all zones)",   iv2)
show("intraday_v2 (R1+S2 only)",  iv2[iv2.zone.isin(['R1','S2'])])

comb = pd.concat([v17a.rename(columns={'pnl65':'pnl_s'}),
                  cam.rename(columns={'pnl65':'pnl_s'}),
                  iv2.rename(columns={'pnl65':'pnl_s'})], ignore_index=True)
# combined quick
all_pnl = list(v17a['pnl65']) + list(cam['pnl65']) + list(iv2['pnl65'])
print(f"  {'COMBINED (all)':<28} {len(v17a)+len(cam)+len(iv2):>4}t | WR {(pd.Series(all_pnl)>0).mean()*100:.1f}% | P&L Rs {sum(all_pnl):>10,.0f}")

# ─────────────────────────────────────────────
# SECTION 2: Conviction sizing
# When v17a + cam on same date same direction → 2x lots
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CONVICTION SIZING (same-day signal confluence)")
print("=" * 65)

v17a['date_s'] = v17a['date'].astype(str)
cam['date_s']  = cam['date'].astype(str)

# cam direction: cam_l3=bull (sell PE), cam_h3=bear (sell CE)
cam['cam_dir'] = cam['strategy'].map({'cam_l3':'bull','cam_h3':'bear'})

# v17a direction
v17a['v17a_dir'] = v17a['opt'].map({'PE':'bull','CE':'bear'})

# merge on date+direction
conf = v17a.merge(
    cam[['date_s','cam_dir','pnl65']].rename(columns={'pnl65':'cam_pnl','cam_dir':'dir2'}),
    left_on=['date_s','v17a_dir'], right_on=['date_s','dir2'], how='left'
)
conf['has_cam'] = conf['cam_pnl'].notna()

solo   = conf[~conf['has_cam']]
double = conf[conf['has_cam']]

print(f"  v17a solo (no cam same day/dir): {len(solo):>3}t | WR {(solo.pnl65>0).mean()*100:.1f}% | Rs {solo.pnl65.sum():,.0f}")
print(f"  v17a + cam confluence (same dir): {len(double):>3}t | WR {(double.pnl65>0).mean()*100:.1f}% | Rs {double.pnl65.sum():,.0f}")

# 2x lot simulation for confluence days
# pnl already at 1-lot; just double it for confluence
conf['adj_pnl'] = conf.apply(
    lambda r: r['pnl65'] * 2 if r['has_cam'] else r['pnl65'], axis=1
)
print(f"\n  Conviction sizing P&L (1x solo, 2x confluence):")
print(f"    Total: Rs {conf['adj_pnl'].sum():,.0f}  vs baseline Rs {v17a['pnl65'].sum():,.0f}")
eq_conv = conf.sort_values('date')['adj_pnl'].cumsum()
dd_conv = eq_conv - eq_conv.cummax()
print(f"    Max DD: Rs {dd_conv.min():,.0f}")

# ─────────────────────────────────────────────
# SECTION 3: CPR + Camarilla alignment concept
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CPR + CAMARILLA ALIGNMENT (54: H3/L3 inside CPR)")
print("=" * 65)

# Best configs already identified
# L3_in_CPR: ITM1, tgt=0.3, sl=1.0 → 635t, 63.3%, Rs+1.8L
# H3_in_CPR: OTM1, tgt=0.3, sl=2.0 → 271t, 57.6%, Rs+1.8L

l3 = df_cam54[(df_cam54.setup=='L3_in_CPR') & (df_cam54.strike_type=='ITM1') &
              (df_cam54.target_pct==0.3) & (df_cam54.sl_pct==1.0)]
h3 = df_cam54[(df_cam54.setup=='H3_in_CPR') & (df_cam54.strike_type=='OTM1') &
              (df_cam54.target_pct==0.3) & (df_cam54.sl_pct==2.0)]

l3_65 = l3['pnl'] * SCALE
h3_65 = h3['pnl'] * SCALE

print(f"  L3 inside CPR → Sell PE:  {len(l3):>4}t | WR {(l3.pnl>0).mean()*100:.1f}% | Rs {l3_65.sum():,.0f}")
print(f"  H3 inside CPR → Sell CE:  {len(h3):>4}t | WR {(h3.pnl>0).mean()*100:.1f}% | Rs {h3_65.sum():,.0f}")

# ─────────────────────────────────────────────
# SECTION 4: Camarilla Touch (55)
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  CAMARILLA INTRADAY TOUCH (55: spot touches H3/L3)")
print("=" * 65)

l3t = df_cam55[(df_cam55.setup=='L3_touch') & (df_cam55.strike_type=='ITM1') &
               (df_cam55.target_pct==0.2) & (df_cam55.sl_pct==0.5)]
h3t = df_cam55[(df_cam55.setup=='H3_touch') & (df_cam55.strike_type=='OTM1') &
               (df_cam55.target_pct==0.5) & (df_cam55.sl_pct==1.0)]

l3t_65 = l3t['pnl'] * SCALE
h3t_65 = h3t['pnl'] * SCALE

print(f"  L3 touch → Sell PE (intraday): {len(l3t):>3}t | WR {(l3t.pnl>0).mean()*100:.1f}% | Rs {l3t_65.sum():,.0f}")
print(f"  H3 touch → Sell CE (intraday): {len(h3t):>3}t | WR {(h3t.pnl>0).mean()*100:.1f}% | Rs {h3t_65.sum():,.0f}")

# ─────────────────────────────────────────────
# SECTION 5: Complete combined picture
# v17a + cam_touch (L3+H3) + iv2(R1+S2 only)
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  COMPLETE COMBINED — RECOMMENDED SETUP (LOT=65)")
print("=" * 65)

iv2_filtered = iv2[iv2.zone.isin(['R1','S2'])].copy()

components = [
    ("v17a",               v17a['pnl65'].values,          v17a['date'].values),
    ("cam_l3",             cam[cam.strategy=='cam_l3']['pnl65'].values,
                           cam[cam.strategy=='cam_l3']['date'].values),
    ("cam_h3",             cam[cam.strategy=='cam_h3']['pnl65'].values,
                           cam[cam.strategy=='cam_h3']['date'].values),
    ("iv2 R1+S2",          iv2_filtered['pnl65'].values,
                           pd.to_datetime(iv2_filtered['date']).values),
]

all_dates, all_pnls = [], []
for name, pnls, dates in components:
    n   = len(pnls)
    wr  = r2((pnls > 0).mean() * 100)
    tot = r2(pnls.sum())
    print(f"  {name:<20} {n:>4}t | WR {wr:>5.1f}% | Rs {tot:>10,.0f}")
    all_dates.extend(dates)
    all_pnls.extend(pnls)

total_pnl = r2(sum(all_pnls))
total_wr  = r2((pd.Series(all_pnls) > 0).mean() * 100)
print(f"  {'─'*55}")
print(f"  {'TOTAL':<20} {len(all_pnls):>4}t | WR {total_wr:>5.1f}% | Rs {total_pnl:>10,.0f}")

# Per-year breakdown
all_df = pd.DataFrame({'date': pd.to_datetime(all_dates), 'pnl': all_pnls})
all_df['year'] = all_df['date'].dt.year
print(f"\n  Year-wise:")
for yr, g in all_df.groupby('year'):
    print(f"    {yr}  {len(g):>3}t | WR {(g.pnl>0).mean()*100:.1f}% | Rs {g.pnl.sum():>9,.0f}")

# Max DD
all_df_s = all_df.sort_values('date').reset_index(drop=True)
eq_all   = all_df_s['pnl'].cumsum()
dd_all   = eq_all - eq_all.cummax()
print(f"\n  Max Drawdown: Rs {dd_all.min():,.0f}")
print(f"  Best year P&L: Rs {all_df.groupby('year').pnl.sum().max():,.0f}")

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────

# Chart 1: Individual equity lines
def eq_line(label, color, pnls, dates):
    df_s = pd.DataFrame({'date': pd.to_datetime(dates), 'pnl': pnls}).sort_values('date')
    eq   = df_s['pnl'].cumsum().values
    return {
        'id': label.replace(' ','_').replace('+',''),
        'label': label, 'color': color,
        'data': [{'time': int(pd.Timestamp(d).timestamp()), 'value': round(float(e), 2)}
                 for d, e in zip(df_s['date'], eq)]
    }

lines = [
    eq_line('v17a',       '#26a69a', v17a['pnl65'].values, v17a['date'].values),
    eq_line('cam_l3',     '#4BC0C0', cam[cam.strategy=='cam_l3']['pnl65'].values, cam[cam.strategy=='cam_l3']['date'].values),
    eq_line('cam_h3',     '#f59e0b', cam[cam.strategy=='cam_h3']['pnl65'].values, cam[cam.strategy=='cam_h3']['date'].values),
    eq_line('iv2 R1+S2',  '#a78bfa', iv2_filtered['pnl65'].values, pd.to_datetime(iv2_filtered['date']).values),
    eq_line('COMBINED',   '#00C853', all_pnls, all_dates),
]
tv1 = {'lines': lines, 'candlestick': [], 'volume': [], 'isTvFormat': False}
send_custom_chart('full_equity', tv1,
    title=f'All Strategies — LOT=65 | {len(all_pnls)}t | WR {total_wr}% | Rs {total_pnl:,.0f}')
print("\n📊 Full equity chart sent")

# Chart 2: Year-wise combined P&L bar
yr_stats = all_df.groupby('year').agg(pnl=('pnl','sum'), n=('pnl','count')).reset_index()
ts_base  = int(pd.Timestamp('2018-01-01').timestamp())
bar_yr   = [{'time': ts_base + i*86400*365,
             'value': round(float(r.pnl),2),
             'color': '#26a69a' if r.pnl>=0 else '#ef5350',
             'label': str(int(r.year))} for i, r in yr_stats.iterrows()]
tv2 = {'lines': [{'id':'yr','label':'Year P&L','seriesType':'bar',
                  'data': bar_yr, 'xLabels': [b['label'] for b in bar_yr]}],
       'candlestick':[], 'volume':[], 'isTvFormat': False}
send_custom_chart('year_pnl', tv2, title='Combined Year-wise P&L (LOT=65)')
print("📊 Year-wise chart sent")

# Chart 3: Conviction sizing vs baseline equity
v17a_s = v17a.sort_values('date').reset_index(drop=True)
base_eq = v17a_s['pnl65'].cumsum()
conv_eq = v17a_s.sort_values('date').reset_index(drop=True)['pnl65'].values
conv_eq_adj = conf.sort_values('date').reset_index(drop=True)['adj_pnl'].cumsum()

tv3 = {'lines': [
    {'id':'base','label':'v17a 1x lot','color':'#8b949e',
     'data': [{'time':int(pd.Timestamp(r.date).timestamp()),'value':round(base_eq[i],2)}
              for i,(_, r) in enumerate(v17a_s.iterrows())]},
    {'id':'conv','label':'v17a conviction (2x on conf)','color':'#26a69a',
     'data': [{'time':int(pd.Timestamp(r['date']).timestamp()),'value':round(float(v),2)}
              for (_, r), v in zip(conf.sort_values('date').iterrows(), conv_eq_adj)]},
], 'candlestick':[], 'volume':[], 'isTvFormat': False}
send_custom_chart('conviction_equity', tv3,
    title='v17a — Conviction Sizing vs Baseline (2x on cam confluence)')
print("📊 Conviction sizing chart sent")

# Chart 4: Strategy contribution bar
contrib = [
    ('v17a',      r2(v17a['pnl65'].sum())),
    ('cam_l3',    r2(cam[cam.strategy=='cam_l3']['pnl65'].sum())),
    ('cam_h3',    r2(cam[cam.strategy=='cam_h3']['pnl65'].sum())),
    ('iv2 R1+S2', r2(iv2_filtered['pnl65'].sum())),
]
ts = int(pd.Timestamp('2021-01-01').timestamp())
bar_c = [{'time': ts+i*86400*180, 'value': v,
          'color':'#26a69a' if v>=0 else '#ef5350',
          'label': n} for i,(n,v) in enumerate(contrib)]
tv4 = {'lines': [{'id':'contrib','label':'Strategy P&L','seriesType':'bar',
                  'data': bar_c, 'xLabels': [b['label'] for b in bar_c]}],
       'candlestick':[], 'volume':[], 'isTvFormat': False}
send_custom_chart('strategy_contrib', tv4, title='Strategy P&L Contribution (LOT=65)')
print("📊 Contribution chart sent")

print("\nAll done!")
