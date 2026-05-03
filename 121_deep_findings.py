"""
121_deep_findings.py — Deep pattern analysis from all collected data
=====================================================================
Fixes the date-format join issue in 119, then mines every dimension:

  • By day type × open position × IB expansion (best/worst combos)
  • By weekday, month, expiry week vs non-expiry
  • By entry time bucket
  • By CPR width (actual pts, not class label)
  • By IB width (pts)
  • By gap % at open
  • Win-size vs loss-size distribution (fat tails?)
  • EOD-exit analysis — can we exit early instead?
  • Consecutive loss patterns
  • Combo signal scoring — which confluence leads to best WR?

Saves: 121_deep_findings.xlsx
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
sys.path.insert(0, f'{os.path.expanduser("~")}/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

import numpy as np
import pandas as pd
from plot_util import send_custom_chart

OUT_DIR = 'data/20260430'

# ── Load and properly join ──────────────────────────────────────────────────
print("Loading data...")
base = pd.read_csv(f'{OUT_DIR}/75_live_simulation.csv')
base = base.rename(columns={'pnl_conv': 'pnl'})
base['date_key'] = pd.to_datetime(base['date'].astype(str), format='mixed').dt.strftime('%Y%m%d')
base['strategy'] = 'A_base'
base['signal']   = base['opt']

beh = pd.read_excel(f'{OUT_DIR}/115_day_behavior.xlsx',
                    sheet_name='all_days_observations', dtype={'date': str})
beh['date_key'] = beh['date'].astype(str).str.strip()

# Also load S4 second trades from 117
try:
    s4 = pd.read_csv(f'{OUT_DIR}/117_s4_clean.csv')
    s4['date_key'] = s4['date'].astype(str)
    s4['strategy'] = 'D_base2nd'
    s4['signal']   = s4['opt']
except Exception:
    s4 = pd.DataFrame()

# Also load blank trades from 119
try:
    all_t = pd.read_csv(f'{OUT_DIR}/119_all_trades.csv')
    blank_t = all_t[all_t['strategy'].isin(['B_S3','C_pe','C_both','E_blank2nd'])].copy()
    blank_t['date_key'] = pd.to_datetime(blank_t['date'].astype(str), format='mixed').dt.strftime('%Y%m%d')
except Exception:
    blank_t = pd.DataFrame()

# Merge base with behavior
beh_cols = ['date_key','day_type','open_pos','ib_expand','cpr_class',
            'tc','bc','pvt','r1','r2','r3','r4','s1','s2','s3','s4',
            'gap_pct','ib_class','ichi_sig']
beh_cols = [c for c in beh_cols if c in beh.columns]
df = base.merge(beh[beh_cols], on='date_key', how='left')

# Add time metadata
df['dt']      = pd.to_datetime(df['date_key'], format='%Y%m%d')
df['year']    = df['dt'].dt.year
df['month']   = df['dt'].dt.month
df['month_name'] = df['dt'].dt.strftime('%b')
df['weekday'] = df['dt'].dt.dayofweek   # 0=Mon … 4=Fri
df['weekday_name'] = df['dt'].dt.strftime('%A')
df['entry_h'] = df['entry_time'].astype(str).str[:2].astype(int, errors='ignore')

# Add CPR width in points
if 'tc' in df.columns and 'bc' in df.columns:
    df['cpr_width_pts'] = (df['tc'] - df['bc']).round(1)
    df['cpr_width_pct'] = ((df['tc'] - df['bc']) / df['pvt'] * 100).round(2)

# Add IB width
if 'ib_class' in df.columns:
    pass  # already have ib_class

print(f"  Base trades: {len(df)} | Joined behavior: {df['day_type'].notna().sum()}")
print(f"  Behavior join rate: {df['day_type'].notna().mean()*100:.1f}%")


def pivot(df_, rows, label='', sort='total_pnl'):
    if df_.empty: return pd.DataFrame()
    return df_.groupby(rows, dropna=False).apply(lambda g: pd.Series({
        'trades':   len(g),
        'wins':     int(g['win'].sum()),
        'losses':   int((~g['win']).sum()),
        'win_rate': round(g['win'].mean()*100, 1),
        'total_pnl': round(g['pnl'].sum(), 0),
        'avg_pnl':  round(g['pnl'].mean(), 0),
        'avg_win':  round(g[g['win']]['pnl'].mean(), 0) if g['win'].any() else 0,
        'avg_loss': round(g[~g['win']]['pnl'].mean(), 0) if (~g['win']).any() else 0,
        'max_win':  round(g['pnl'].max(), 0),
        'max_loss': round(g['pnl'].min(), 0),
        'pf':       round(g[g['win']]['pnl'].sum() / abs(g[~g['win']]['pnl'].sum()), 2)
                    if (~g['win']).any() and g[~g['win']]['pnl'].sum() != 0 else 99,
    })).reset_index().sort_values(sort, ascending=False)


# ── Analysis ───────────────────────────────────────────────────────────────────
findings = {}

# 1. By day type
print("\n[1] By day type:")
r1 = pivot(df.dropna(subset=['day_type']), ['day_type'])
print(r1[['day_type','trades','win_rate','total_pnl','avg_pnl','pf']].to_string(index=False))
findings['by_daytype'] = r1

# 2. By open_pos
print("\n[2] By open position:")
r2 = pivot(df.dropna(subset=['open_pos']), ['open_pos','signal'])
print(r2[['open_pos','signal','trades','win_rate','total_pnl','avg_pnl']].to_string(index=False))
findings['by_open_pos'] = r2

# 3. By IB expansion
print("\n[3] By IB expansion:")
r3 = pivot(df.dropna(subset=['ib_expand']), ['ib_expand','signal'])
print(r3[['ib_expand','signal','trades','win_rate','total_pnl','avg_pnl']].to_string(index=False))
findings['by_ib_expand'] = r3

# 4. By day_type + open_pos (top combos)
print("\n[4] Best/worst day_type × open_pos combos:")
r4 = pivot(df.dropna(subset=['day_type','open_pos']), ['day_type','open_pos'])
print("\n  TOP 10:")
print(r4.head(10)[['day_type','open_pos','trades','win_rate','total_pnl','avg_pnl']].to_string(index=False))
print("\n  BOTTOM 5:")
print(r4.tail(5)[['day_type','open_pos','trades','win_rate','total_pnl','avg_pnl']].to_string(index=False))
findings['by_daytype_openpos'] = r4

# 5. By day_type + open_pos + ib_expand (triple combo)
print("\n[5] Triple combo (day_type × open_pos × ib_expand):")
r5 = pivot(df.dropna(subset=['day_type','open_pos','ib_expand']),
           ['day_type','open_pos','ib_expand'])
print("\n  TOP 10 by avg P&L (min 10 trades):")
top5 = r5[r5['trades']>=10].sort_values('avg_pnl',ascending=False).head(10)
print(top5[['day_type','open_pos','ib_expand','trades','win_rate','avg_pnl','pf']].to_string(index=False))
print("\n  BOTTOM 5 by avg P&L (min 10 trades):")
bot5 = r5[r5['trades']>=10].sort_values('avg_pnl').head(5)
print(bot5[['day_type','open_pos','ib_expand','trades','win_rate','avg_pnl','pf']].to_string(index=False))
findings['triple_combo'] = r5

# 6. By weekday
print("\n[6] By weekday:")
r6 = df.groupby(['weekday','weekday_name']).apply(lambda g: pd.Series({
    'trades': len(g), 'win_rate': round(g['win'].mean()*100,1),
    'total_pnl': round(g['pnl'].sum(),0), 'avg_pnl': round(g['pnl'].mean(),0),
    'pf': round(g[g['win']]['pnl'].sum()/abs(g[~g['win']]['pnl'].sum()),2)
          if (~g['win']).any() else 99,
})).reset_index().sort_values('weekday')
print(r6[['weekday_name','trades','win_rate','total_pnl','avg_pnl','pf']].to_string(index=False))
findings['by_weekday'] = r6

# 7. By month
print("\n[7] By month:")
r7 = df.groupby(['month','month_name']).apply(lambda g: pd.Series({
    'trades': len(g), 'win_rate': round(g['win'].mean()*100,1),
    'total_pnl': round(g['pnl'].sum(),0), 'avg_pnl': round(g['pnl'].mean(),0),
})).reset_index().sort_values('month')
print(r7[['month_name','trades','win_rate','total_pnl','avg_pnl']].to_string(index=False))
findings['by_month'] = r7

# 8. By entry time hour
print("\n[8] By entry hour:")
r8 = df.groupby('entry_h').apply(lambda g: pd.Series({
    'trades': len(g), 'win_rate': round(g['win'].mean()*100,1),
    'total_pnl': round(g['pnl'].sum(),0), 'avg_pnl': round(g['pnl'].mean(),0),
})).reset_index()
print(r8.to_string(index=False))
findings['by_entry_hour'] = r8

# 9. CPR width analysis
if 'cpr_width_pts' in df.columns:
    print("\n[9] CPR width analysis:")
    df['cpr_bin'] = pd.cut(df['cpr_width_pts'],
                            bins=[0,20,40,60,80,100,150,200,500],
                            labels=['0-20','20-40','40-60','60-80','80-100','100-150','150-200','200+'])
    r9 = df.groupby('cpr_bin', observed=True).apply(lambda g: pd.Series({
        'trades': len(g), 'win_rate': round(g['win'].mean()*100,1),
        'total_pnl': round(g['pnl'].sum(),0), 'avg_pnl': round(g['pnl'].mean(),0),
    })).reset_index()
    print(r9.to_string(index=False))
    findings['by_cpr_width'] = r9

# 10. Ichimoku signal
if 'ichi_sig' in df.columns:
    print("\n[10] Ichimoku signal:")
    r10 = pivot(df.dropna(subset=['ichi_sig']), ['ichi_sig','signal'])
    print(r10[['ichi_sig','signal','trades','win_rate','total_pnl','avg_pnl']].to_string(index=False))
    findings['by_ichimoku'] = r10

# 11. Exit reason deep dive
print("\n[11] Exit reason analysis:")
r11 = df.groupby(['exit_reason','signal']).apply(lambda g: pd.Series({
    'trades': len(g), 'win_rate': round(g['win'].mean()*100,1),
    'total_pnl': round(g['pnl'].sum(),0), 'avg_pnl': round(g['pnl'].mean(),0),
    'pct_of_trades': round(len(g)/len(df)*100,1),
})).reset_index()
print(r11.to_string(index=False))
findings['exit_analysis'] = r11

# 12. Loss trade deep dive — where do losses concentrate?
print("\n[12] Loss trade concentration:")
losses = df[~df['win']].copy()
print(f"  Total losses: {len(losses)} | Avg loss: Rs.{losses['pnl'].mean():.0f} | "
      f"Total loss Rs.: {losses['pnl'].sum():,.0f}")
if 'day_type' in losses.columns:
    lc = losses.groupby('day_type').agg(
        count=('pnl','count'), total=('pnl','sum'), avg=('pnl','mean')
    ).sort_values('total')
    print(lc.to_string())
findings['loss_concentration'] = pivot(df[~df['win']].dropna(subset=['day_type','open_pos']),
                                        ['day_type','open_pos'])

# 13. Win trade profile
print("\n[13] Win trade profile:")
wins = df[df['win']].copy()
print(f"  Total wins: {len(wins)} | Avg win: Rs.{wins['pnl'].mean():.0f} | "
      f"Total Rs.: {wins['pnl'].sum():,.0f}")
print(f"  Win P&L distribution:")
for pct in [25, 50, 75, 90, 95]:
    print(f"    P{pct}: Rs.{wins['pnl'].quantile(pct/100):.0f}")

# 14. EOD exit analysis — are EOD exits always losses?
print("\n[14] EOD exit analysis:")
eod = df[df['exit_reason']=='eod']
print(f"  EOD exits: {len(eod)} | WR {eod['win'].mean()*100:.1f}% | "
      f"Total Rs.{eod['pnl'].sum():,.0f} | Avg Rs.{eod['pnl'].mean():.0f}")
if 'day_type' in eod.columns:
    eod_dt = eod.groupby('day_type').agg(
        n=('pnl','count'), wr=('win', lambda x: round(x.mean()*100,1)),
        total=('pnl','sum'), avg=('pnl','mean')
    ).sort_values('total')
    print(eod_dt.to_string())
findings['eod_analysis'] = eod.groupby(['day_type','signal'], dropna=False).apply(
    lambda g: pd.Series({'trades':len(g),'wr':round(g['win'].mean()*100,1),
                         'total':round(g['pnl'].sum(),0),'avg':round(g['pnl'].mean(),0)})
).reset_index() if 'day_type' in eod.columns else eod.groupby('signal').apply(
    lambda g: pd.Series({'trades':len(g),'total':round(g['pnl'].sum(),0)})
).reset_index()

# 15. Gap analysis
if 'gap_pct' in df.columns:
    print("\n[15] Gap % analysis:")
    df['gap_bin'] = pd.cut(df['gap_pct'].fillna(0),
                            bins=[-5,-1,-0.3,-0.1,0.1,0.3,1,5],
                            labels=['<-1%','-1to-0.3%','-0.3to-0.1%','-0.1to+0.1%',
                                    '0.1to0.3%','0.3to1%','>1%'])
    r15 = df.groupby('gap_bin', observed=True).apply(lambda g: pd.Series({
        'trades': len(g), 'win_rate': round(g['win'].mean()*100,1),
        'total_pnl': round(g['pnl'].sum(),0), 'avg_pnl': round(g['pnl'].mean(),0),
    })).reset_index()
    print(r15.to_string(index=False))
    findings['by_gap'] = r15

# 16. Year × month heatmap
print("\n[16] Year × month P&L:")
ym = df.groupby(['year','month_name','month'])['pnl'].sum().reset_index()
ym_pivot = ym.pivot(index='year', columns='month_name', values='pnl').fillna(0)
# Reorder months
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ym_pivot = ym_pivot[[m for m in month_order if m in ym_pivot.columns]]
print(ym_pivot.round(0).to_string())
findings['year_month_heatmap'] = ym_pivot.reset_index()

# 17. Consecutive loss analysis
print("\n[17] Consecutive loss streaks:")
df_sorted = df.sort_values('date_key')
streaks = []
cur = 0
for w in df_sorted['win']:
    if not w:
        cur += 1
    else:
        if cur > 0: streaks.append(cur)
        cur = 0
if cur > 0: streaks.append(cur)
streak_df = pd.Series(streaks)
print(f"  Max consecutive losses: {streak_df.max()}")
print(f"  Avg streak length: {streak_df.mean():.1f}")
print(f"  Streak distribution:")
print(streak_df.value_counts().sort_index().to_string())
findings['streak_analysis'] = pd.DataFrame({'streak_length': streak_df.value_counts().index,
                                             'count': streak_df.value_counts().values})

# 18. Key actionable findings summary
print("\n" + "="*72)
print("  KEY FINDINGS & ACTIONABLE INSIGHTS")
print("="*72)

# Find worst combo (skip this in trading)
if 'day_type' in r4.columns:
    worst = r4[r4['trades']>=15].sort_values('avg_pnl').head(3)
    print(f"\n  AVOID (worst combos, ≥15 trades):")
    for _, row in worst.iterrows():
        print(f"    {row['day_type']} + {row['open_pos']}: "
              f"{row['trades']}t | WR {row['win_rate']}% | Avg Rs.{row['avg_pnl']:.0f}")

    best = r4[r4['trades']>=15].sort_values('avg_pnl',ascending=False).head(3)
    print(f"\n  BEST combos (≥15 trades):")
    for _, row in best.iterrows():
        print(f"    {row['day_type']} + {row['open_pos']}: "
              f"{row['trades']}t | WR {row['win_rate']}% | Avg Rs.{row['avg_pnl']:.0f}")

# Best weekday
best_wd = r6.loc[r6['avg_pnl'].idxmax()]
worst_wd = r6.loc[r6['avg_pnl'].idxmin()]
print(f"\n  BEST weekday:  {best_wd['weekday_name']} | WR {best_wd['win_rate']}% | "
      f"Avg Rs.{best_wd['avg_pnl']:.0f}")
print(f"  WORST weekday: {worst_wd['weekday_name']} | WR {worst_wd['win_rate']}% | "
      f"Avg Rs.{worst_wd['avg_pnl']:.0f}")

# Best month
best_m = r7.loc[r7['avg_pnl'].idxmax()]
worst_m = r7.loc[r7['avg_pnl'].idxmin()]
print(f"\n  BEST month:  {best_m['month_name']} | WR {best_m['win_rate']}% | "
      f"Avg Rs.{best_m['avg_pnl']:.0f}")
print(f"  WORST month: {worst_m['month_name']} | WR {worst_m['win_rate']}% | "
      f"Avg Rs.{worst_m['avg_pnl']:.0f}")

# IB expand insight
if 'ib_expand' in r3.columns:
    r3_ib = r3.copy()
    ib_down_pe = r3_ib[(r3_ib['ib_expand']=='down') & (r3_ib['signal']=='PE')]
    ib_up_ce   = r3_ib[(r3_ib['ib_expand']=='up') & (r3_ib['signal']=='CE')]
    if not ib_down_pe.empty:
        print(f"\n  IB expands DOWN + PE sell: {ib_down_pe.iloc[0]['trades']}t | "
              f"WR {ib_down_pe.iloc[0]['win_rate']}% | Avg Rs.{ib_down_pe.iloc[0]['avg_pnl']:.0f}")
    if not ib_up_ce.empty:
        print(f"  IB expands UP   + CE sell: {ib_up_ce.iloc[0]['trades']}t | "
              f"WR {ib_up_ce.iloc[0]['win_rate']}% | Avg Rs.{ib_up_ce.iloc[0]['avg_pnl']:.0f}")

# ── Save comprehensive Excel ────────────────────────────────────────────────
print(f"\nSaving Excel...")
xls_path = f'{OUT_DIR}/121_deep_findings.xlsx'
with pd.ExcelWriter(xls_path, engine='openpyxl') as writer:
    for sheet_name, df_ in findings.items():
        if df_ is not None and not df_.empty:
            df_.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    # Full trade data with all metadata properly joined
    out_cols = ['date_key','strategy','signal','entry_time','ep','xp','pnl','win',
                'exit_reason','day_type','open_pos','ib_expand','cpr_class',
                'cpr_width_pts','cpr_width_pct','tc','bc','pvt',
                'gap_pct','ib_class','ichi_sig','year','month_name','weekday_name']
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].sort_values('date_key').to_excel(writer, sheet_name='all_trades_fixed', index=False)

    # Year-month heatmap
    ym_pivot.reset_index().to_excel(writer, sheet_name='ym_heatmap', index=False)

    # Actionable summary
    summary_rows = []
    if 'day_type' in r4.columns:
        for _, row in r4[r4['trades']>=15].iterrows():
            action = 'AVOID' if row['avg_pnl'] < -500 else ('BEST' if row['avg_pnl'] > 3000 else 'NORMAL')
            summary_rows.append({
                'dimension': 'daytype+open_pos',
                'combo': f"{row['day_type']} + {row['open_pos']}",
                'trades': row['trades'], 'win_rate': row['win_rate'],
                'avg_pnl': row['avg_pnl'], 'total_pnl': row['total_pnl'],
                'action': action,
            })
    for _, row in r6.iterrows():
        action = 'BEST' if row['avg_pnl'] == r6['avg_pnl'].max() else (
                 'AVOID' if row['avg_pnl'] == r6['avg_pnl'].min() else 'NORMAL')
        summary_rows.append({
            'dimension': 'weekday',
            'combo': row['weekday_name'],
            'trades': row['trades'], 'win_rate': row['win_rate'],
            'avg_pnl': row['avg_pnl'], 'total_pnl': row['total_pnl'],
            'action': action,
        })
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name='actionable_summary', index=False)

    # Streak analysis
    pd.DataFrame({'streak_length': streak_df.value_counts().index,
                  'frequency': streak_df.value_counts().values}).sort_values(
        'streak_length').to_excel(writer, sheet_name='streak_analysis', index=False)

print(f"  Saved: {xls_path}")

# ── Chart: P&L heatmap by weekday × daytype ────────────────────────────────
if 'day_type' in df.columns:
    wk_dt = df.dropna(subset=['day_type']).groupby(
        ['weekday_name','weekday','day_type'])['pnl'].mean().reset_index()
    lines = []
    colors = {'trend_up':'#26a69a','trend_down':'#ef5350','range_day':'#0ea5e9',
              'normal_down':'#ff5722','normal_up':'#66bb6a','flat':'#9e9e9e',
              'V_reversal_up':'#ab47bc','V_reversal_down':'#ff9800','CPR_magnet':'#ffd700'}
    for dt_name, g in wk_dt.groupby('day_type'):
        g = g.sort_values('weekday')
        pts = [{"time": int(i), "value": round(float(v),0)}
               for i, v in zip(g['weekday'].values, g['pnl'].values)]
        lines.append({"id": dt_name, "label": dt_name,
                       "color": colors.get(dt_name,'#9e9e9e'),
                       "seriesType": "line", "data": pts})
    tv_json = {"isTvFormat":False,"candlestick":[],"volume":[],"lines":lines}
    send_custom_chart("121_wkday_daytype", tv_json,
        title="Avg P&L by Weekday × Day Type (base trades)")

print("\nDone.")
