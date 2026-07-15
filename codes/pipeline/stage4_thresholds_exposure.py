# %% [markdown]
# # Stage 4 — Heat/Cold Stress Thresholds & Cumulative Exposure  (proposal aim 5)
#
# Loads `analysis_base.parquet` + `farm_daily_anom.parquet` (Stage 1). ANALYSIS-ONLY.
#
# 1. Trailing-window heat exposure: mean THI anomaly over prior W days (per farm).
# 2. Cumulative-exposure sweep: within-animal effect vs window length, best window per trait,
#    raw vs DIM-controlled (the SCS signal that was faint same-day in Stage 3 should build up).
# 3. Heat-wave / cold-snap events (Petrocchi/Matera thresholds): frequency + within-animal effect.
# 4. Threshold estimation: where each trait inflects with THI (summer, within-animal).
# 5. Regional heterogeneity: heat sensitivity by macro-area and Campania+Lazio vs Rest.
#
# Rules: R1 anomaly, R3 cumulative, R4 DIM+parity control, R5 within-animal + farm-cluster SE.
# Anchors: Petrocchi 2023 (event thresholds), XGBoost heat paper (cumulative), Matera 2022.

# %%
import pandas as pd, numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def find_root():
    cands=[Path("Thesis_Data"), Path("../../Thesis_Data")]
    try: cands.append(Path(__file__).resolve().parents[2]/"Thesis_Data")
    except NameError: pass
    for c in cands:
        if c.exists(): return c.resolve().parent
    raise SystemExit("no Thesis_Data")
ROOT=find_root(); OUT=ROOT/"codes"/"pipeline"/"outputs"
FIG=OUT/"figures"; TAB=OUT/"tables"; FIG.mkdir(exist_ok=True,parents=True); TAB.mkdir(exist_ok=True,parents=True)
def rule(m): print("\n"+"="*78+f"\n{m}\n"+"="*78)
def verdict(n,ok,d=""): print(f"  [{'MATCH' if ok else 'CHECK'}] {n}"+(f"  --  {d}" if d else ""))

base = pd.read_parquet(OUT/"analysis_base.parquet")
daily= pd.read_parquet(OUT/"farm_daily_anom.parquet").sort_values(["Farm_Code","date"]).reset_index(drop=True)
WIN=[1,3,7,14,30,45,60,90]
rule("LOADED"); print(f"  test-days={len(base):,} | daily weather rows={len(daily):,}")

# %% [markdown]
# ## 4.1 Trailing-window exposure (mean THI anomaly over prior W days, per farm)
# Weather is daily-complete per farm, so an integer W-row window == a W-day calendar window.

# %%
rule("4.1  TRAILING-WINDOW EXPOSURE")
for W in WIN:
    daily[f"anom_{W}"]=(daily.groupby("Farm_Code")["THI_anom"]
                        .rolling(W, min_periods=min(W,max(1,W//2))).mean().reset_index(level=0,drop=True))
wcols=[f"anom_{W}" for W in WIN]
b = base.merge(daily[["Farm_Code","date"]+wcols], left_on=["Farm_Code","dtt"], right_on=["Farm_Code","date"], how="left").drop(columns="date")
print(f"  merged windows onto test-days; coverage anom_30 = {b['anom_30'].notna().mean():.3%}")

# %% [markdown]
# ## 4.2 Cumulative-exposure sweep (within-animal, summer)
# For each trait and window W: within-animal FE of the trait on the windowed anomaly, DIM+parity
# controlled, farm-clustered SE. Milk & SCS also get a NO-DIM-control run to show how much the
# (uncontrolled) SCS effect is inflated by lactation stage.

# %%
rule("4.2  CUMULATIVE-EXPOSURE SWEEP")
S = b[b.is_summer_warmhalf].copy()
S["DIM2"]=S.DIM.astype(float)**2
grp=S.groupby("Animal_ID")
# precompute demeaned controls (independent of W)
for c in ["DIM","DIM2","parity"]:
    S[c+"_w"]=S[c]-grp[c].transform("mean")
def fe_window(trait, W, dim_control=True):
    cols=[trait,f"anom_{W}","DIM_w","DIM2_w","parity_w","Animal_ID","Farm_Code"]
    d=S[cols].dropna().copy()
    d[trait+"_w"]=d[trait]-d.groupby("Animal_ID")[trait].transform("mean")
    d["x_w"]=d[f"anom_{W}"]-d.groupby("Animal_ID")[f"anom_{W}"].transform("mean")
    rhs = "x_w + DIM_w + DIM2_w + parity_w - 1" if dim_control else "x_w - 1"
    m=smf.ols(f"{trait}_w ~ {rhs}", data=d).fit(cov_type="cluster",cov_kwds={"groups":d.Farm_Code})
    return 10*m.params["x_w"], m.pvalues["x_w"]
rows=[]
for t in ["milk_kg","SCS","fat_p","protein_p"]:
    for W in WIN:
        per10,p=fe_window(t,W,True); rows.append({"trait":t,"window":W,"per10_DIMctrl":round(per10,4),"p":float(p)})
prof=pd.DataFrame(rows); prof.to_csv(TAB/"exposure_window_profile.csv",index=False)
piv=prof.pivot(index="window",columns="trait",values="per10_DIMctrl")
print("  effect per +10 THI-anomaly by window (DIM-controlled):"); print(piv.round(4).to_string())
# raw vs controlled for SCS (and milk)
print("\n  SCS raw vs DIM-controlled (per +10 anomaly):")
scs_cmp=[]
for W in WIN:
    r,_=fe_window("SCS",W,False); c,_=fe_window("SCS",W,True); scs_cmp.append({"window":W,"SCS_raw":round(r,4),"SCS_DIMctrl":round(c,4)})
print(pd.DataFrame(scs_cmp).to_string(index=False))
# best DEFENSIBLE window per trait: restrict to <=45 d; 60-90 d are collinear with season/DIM
# (a 90-d trailing window for a summer test-day reaches back into spring), so their larger
# effects are partly seasonal leakage, not pure cumulative heat.
best={}
DEFENSIBLE=45
for t in ["milk_kg","SCS","fat_p","protein_p"]:
    pool=prof[(prof.trait==t)&(prof.window<=DEFENSIBLE)]
    sig=pool[pool.p<0.05]
    src=sig if len(sig) else pool
    pick=src.iloc[src.per10_DIMctrl.abs().argmax()]
    best[t]={"window":int(pick.window),"per10":float(pick.per10_DIMctrl),"p":float(pick.p)}
print("\n  best DEFENSIBLE window (<=45d) per trait:",
      {k:(v['window'], round(v['per10'],3), f'p={v["p"]:.1e}') for k,v in best.items()})
print("  NOTE: 60-90d windows keep growing (milk 90d=-0.62) but are season/DIM-collinear -> reported, not headline.")

fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ax[0].plot(piv.index,piv["milk_kg"],marker="o",label="milk_kg"); ax[0].axhline(0,c="grey",lw=.8)
ax[0].set_title("Milk: heat effect vs accumulation window"); ax[0].set_xlabel("window (days)"); ax[0].set_ylabel("Δ per +10 anomaly"); ax[0].grid(alpha=.3)
sc=pd.DataFrame(scs_cmp).set_index("window")
ax[1].plot(sc.index,sc.SCS_raw,marker="o",label="SCS (no DIM ctrl)")
ax[1].plot(sc.index,sc.SCS_DIMctrl,marker="s",label="SCS (DIM-controlled)"); ax[1].axhline(0,c="grey",lw=.8)
ax[1].set_title("SCS: cumulative heat, raw vs DIM-controlled"); ax[1].set_xlabel("window (days)"); ax[1].set_ylabel("Δ per +10 anomaly"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG/"exposure_window_profile.png",dpi=130); plt.show()
print(f"  wrote {FIG/'exposure_window_profile.png'}")

# %% [markdown]
# ## 4.3 Heat-wave / cold-snap events (Petrocchi/Matera thresholds)
# Heat wave: THI_1_avg>78, >=3 consecutive days (Petrocchi 2023); severe >82. Cold snap:
# THI_1_avg<50, >=3 d (Matera 2022); severe: temperature_avg<0. Runs do not cross farms.

# %%
rule("4.3  HEAT-WAVE / COLD-SNAP EVENTS")
farm=daily.Farm_Code.values
def in_run(cond, min_len=3):
    b_=cond.values.astype(int)
    change=np.concatenate([[True],(b_[1:]!=b_[:-1])|(farm[1:]!=farm[:-1])])
    runid=np.cumsum(change)
    rl=pd.Series(b_).groupby(runid).transform("size").values
    return (b_==1)&(rl>=min_len)
daily["heat_event"]  =in_run(daily.THI_1_avg>78)
daily["severe_heat"] =in_run(daily.THI_1_avg>82)
daily["cold_snap"]   =in_run(daily.THI_1_avg<50)
daily["severe_cold"] =in_run(daily.temperature_avg<0)
freq={k:round(daily[k].mean()*100,3) for k in ["heat_event","severe_heat","cold_snap","severe_cold"]}
print("  % of farm-days in event:", freq)
ev=base.merge(daily[["Farm_Code","date","heat_event","cold_snap"]], left_on=["Farm_Code","dtt"], right_on=["Farm_Code","date"],how="left").drop(columns="date")
ev["milk_w"]=ev.milk_kg-ev.groupby("Animal_ID").milk_kg.transform("mean")
hw=ev[ev.is_summer_warmhalf]; cs=ev[~ev.is_summer_warmhalf]
d_heat=hw.loc[hw.heat_event==True,"milk_w"].mean()-hw.loc[hw.heat_event==False,"milk_w"].mean()
d_cold=cs.loc[cs.cold_snap==True,"milk_w"].mean()-cs.loc[cs.cold_snap==False,"milk_w"].mean()
print(f"  within-animal milk: heat-wave vs not (summer) = {d_heat:+.3f} kg ; cold-snap vs not (winter) = {d_cold:+.3f} kg")
json.dump({"freq_pct":freq,"milk_delta_heatwave":round(float(d_heat),3),"milk_delta_coldsnap":round(float(d_cold),3)},
          open(TAB/"events_summary.json","w"),indent=2)

# %% [markdown]
# ## 4.4 Threshold estimation — where traits inflect with THI (summer, within-animal)
# In summer season is largely fixed, so the within-animal deviation vs absolute daily-max THI is
# interpretable. Segmented (breakpoint) fit gives the THI where milk starts falling / SCS rising.

# %%
rule("4.4  THRESHOLD ESTIMATION (summer)")
def breakpoint(trait):
    d=b[b.is_summer_warmhalf][[trait,"THI_1_max","Animal_ID"]].dropna().copy()
    d["dev"]=d[trait]-d.groupby("Animal_ID")[trait].transform("mean")
    d["band"]=pd.cut(d.THI_1_max,bins=np.arange(60,92,2))
    cur=d.groupby("band",observed=True).agg(dev=("dev","mean"),n=("dev","size"))
    cur=cur[cur.n>1500].reset_index(); cur["mid"]=cur.band.apply(lambda x:x.mid).astype(float)
    x=cur.mid.values; y=cur.dev.values; best=None
    for bp in np.arange(x.min()+2,x.max()-2,1):
        X=np.column_stack([np.ones_like(x),x,np.maximum(x-bp,0)])
        beta,*_=np.linalg.lstsq(X,y,rcond=None); rss=float(((y-X@beta)**2).sum())
        if best is None or rss<best[1]: best=(bp,rss,beta)
    return best[0],best[2][1],best[2][1]+best[2][2],cur
for t in ["milk_kg","SCS"]:
    bp,s1,s2,cur=breakpoint(t)
    print(f"  {t:8s} breakpoint ≈ THI_max {bp:.0f}  (slope below={s1:+.4f}, above={s2:+.4f} per THI)")
    cur.to_csv(TAB/f"threshold_curve_{t}.csv",index=False)

# %% [markdown]
# ## 4.5 Regional heterogeneity — heat sensitivity (30-day anomaly, summer)

# %%
rule("4.5  REGIONAL HETEROGENEITY")
def region_effect(sub,trait):
    d=sub[[trait,"anom_30","DIM","Animal_ID","Farm_Code"]].dropna().copy()
    if d.Animal_ID.nunique()<200: return None
    d["DIM2"]=d.DIM.astype(float)**2
    g=d.groupby("Animal_ID")
    for c in [trait,"anom_30","DIM","DIM2"]: d[c+"_w"]=d[c]-g[c].transform("mean")
    m=smf.ols(f"{trait}_w ~ anom_30_w + DIM_w + DIM2_w - 1",data=d).fit(cov_type="cluster",cov_kwds={"groups":d.Farm_Code})
    return 10*m.params["anom_30_w"], 10*1.96*m.bse["anom_30_w"], int(d.Animal_ID.nunique())
S30=b[b.is_summer_warmhalf].copy()
groups={a:S30[S30.macro_area==a] for a in ["North","Central","South"]}
groups["Campania+Lazio"]=S30[S30.Region.isin(["Campania","Lazio"])]
groups["Rest"]=S30[~S30.Region.isin(["Campania","Lazio"])]
forest=[]
for name,sub in groups.items():
    for t in ["milk_kg","SCS"]:
        r=region_effect(sub,t)
        if r: forest.append({"group":name,"trait":t,"per10":round(r[0],4),"ci95":round(r[1],4),"animals":r[2]})
fdf=pd.DataFrame(forest); fdf.to_csv(TAB/"regional_forest.csv",index=False)
print(fdf.to_string(index=False))
fig,ax=plt.subplots(1,2,figsize=(12,4))
for j,t in enumerate(["milk_kg","SCS"]):
    ss=fdf[fdf.trait==t]; yv=range(len(ss))
    ax[j].errorbar(ss.per10,yv,xerr=ss.ci95,fmt="o",capsize=3)
    ax[j].axvline(0,c="grey",lw=.8); ax[j].set_yticks(list(yv)); ax[j].set_yticklabels(ss.group)
    ax[j].set_title(f"{t}: Δ per +10 THI-anom (30d), summer"); ax[j].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG/"regional_forest.png",dpi=130); plt.show()
print(f"  wrote {FIG/'regional_forest.png'}")

# %% [markdown]
# ## 4.6 Literature-comparison checkpoint

# %%
rule("4.6  LITERATURE COMPARISON")
verdict("Petrocchi/EDA: cold snaps more frequent than heat waves", freq["cold_snap"]>freq["heat_event"],
        f"cold_snap={freq['cold_snap']}% vs heat_event={freq['heat_event']}% of farm-days")
verdict("Severe heat (THI>82) essentially absent", freq["severe_heat"]<0.1, f"severe_heat={freq['severe_heat']}% of farm-days")
scs_same=abs(prof[(prof.trait=='SCS')&(prof.window==1)].per10_DIMctrl.iloc[0])
scs_best=abs(best['SCS']['per10'])
verdict("XGBoost paper: cumulative heat > same-day (SCS)", scs_best>=scs_same,
        f"SCS |effect| same-day={scs_same:.3f} -> best({best['SCS']['window']}d)={scs_best:.3f}")
verdict("SCS heat effect emerges & is positive at some window",
        prof[(prof.trait=='SCS')&(prof.per10_DIMctrl>0)&(prof.p<0.05)].shape[0]>0,
        f"significant positive-SCS windows: {sorted(prof[(prof.trait=='SCS')&(prof.per10_DIMctrl>0)&(prof.p<0.05)].window.tolist())}")
summary={"best_window":{k:v['window'] for k,v in best.items()},
         "best_per10":{k:round(v['per10'],4) for k,v in best.items()},
         "event_freq_pct":freq,"milk_delta_heatwave":round(float(d_heat),3),"milk_delta_coldsnap":round(float(d_cold),3)}
(TAB/"stage4_summary.json").write_text(json.dumps(summary,indent=2))
rule("STAGE 4 COMPLETE"); print("  summary:",summary)

# %% [markdown]
# ## 4.7 Figures — thresholds & exposure

# %%
rule("4.7  FIGURES")
# Fig 4.2b: full window profile for all traits (DIM-controlled)
fig,ax=plt.subplots(figsize=(9,5))
for t in ["milk_kg","SCS","fat_p","protein_p"]:
    s=prof[prof.trait==t]; ax.plot(s.window,s.per10_DIMctrl,marker="o",label=t)
ax.axhline(0,c="k",lw=.8); ax.axvspan(45,90,color="grey",alpha=.08,label="60-90d season-collinear")
ax.set_xlabel("accumulation window (days)"); ax.set_ylabel("Δ per +10 THI-anomaly")
ax.set_title("4.2  Cumulative heat effect vs window (all traits, DIM-controlled)"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG/"s4_window_profile_all.png",dpi=130); plt.show()

# Fig 4.3: event frequency + within-animal milk effect
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ev_names=list(freq.keys()); ev_vals=[freq[k] for k in ev_names]
ax[0].bar(ev_names,ev_vals,color=["darkorange","red","steelblue","navy"])
ax[0].set_ylabel("% of farm-days"); ax[0].set_title("4.3  Event frequency (cold snaps ≫ heat waves)")
for i,v in enumerate(ev_vals): ax[0].text(i,v+.2,f"{v}%",ha="center",fontsize=8)
ax[0].grid(alpha=.3,axis="y"); ax[0].tick_params(axis="x",rotation=20)
ax[1].bar(["heat wave\n(summer)","cold snap\n(winter)"],[d_heat,d_cold],color=["firebrick","steelblue"])
ax[1].axhline(0,c="k",lw=.8); ax[1].set_ylabel("Δ within-animal milk_kg"); ax[1].set_title("4.3  Within-animal milk in vs out of event")
ax[1].grid(alpha=.3,axis="y")
plt.tight_layout(); plt.savefig(FIG/"s4_events.png",dpi=130); plt.show()

# Fig 4.4: threshold curves (milk & SCS vs THI_max) with breakpoint
def seg(x,y):
    best=None
    for bp in np.arange(x.min()+2,x.max()-2,1):
        X=np.column_stack([np.ones_like(x),x,np.maximum(x-bp,0)]); be,*_=np.linalg.lstsq(X,y,rcond=None)
        rss=float(((y-X@be)**2).sum())
        if best is None or rss<best[1]: best=(bp,rss,be)
    return best
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
for a,t,col in zip(ax,["milk_kg","SCS"],["steelblue","firebrick"]):
    c=pd.read_csv(TAB/f"threshold_curve_{t}.csv"); x=c["mid"].values.astype(float); y=c["dev"].values
    bp,_,be=seg(x,y); xg=np.linspace(x.min(),x.max(),100); yh=be[0]+be[1]*xg+be[2]*np.maximum(xg-bp,0)
    a.scatter(x,y,s=25,color=col); a.plot(xg,yh,c="k"); a.axvline(bp,c="grey",ls="--")
    a.axhline(0,c="grey",lw=.6); a.set_title(f"4.4  {t}: within-animal dev vs THI_max (bp≈{bp:.0f})")
    a.set_xlabel("THI_1_max (summer)"); a.set_ylabel(f"Δ {t}"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG/"s4_thresholds.png",dpi=130); plt.show()
print("  wrote s4_window_profile_all / s4_events / s4_thresholds")
