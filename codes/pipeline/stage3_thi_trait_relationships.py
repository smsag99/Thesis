# %% [markdown]
# # Stage 3 — Climate Signal & THI–Trait Relationships
#
# The heart of the thesis. Loads `outputs/analysis_base.parquet`. ANALYSIS-ONLY.
#
# 1. Season-stratified correlations (the season confound: annual r cancels).
# 2. THI-class response curves (raw vs within-animal), find the apparent optimum.
# 3. Positive control — Matera 2022 model: THI-class × parity, SOL × parity, year-season of
#    calving, + LR covariate; test whether THI×parity is significant (Matera's key result).
# 4. Causal within-animal anomaly model (R1/R4/R5): the de-confounded heat effect, summer.
# 5. Non-linear shape: quadratic THI + breakpoint for the SCS heat threshold.
#
# Two THI measures are used deliberately: **raw THI_1_avg** to reproduce Matera's method, and
# the **THI anomaly** (THI − farm-month normal) for the causal claim. Comparing them shows the
# season confound. Anchors: Matera 2022, Piscopo 2024.

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

df = pd.read_parquet(OUT/"analysis_base.parquet")
df["DIM2"] = df.DIM.astype(float)**2
df["ys_calv"] = df.dtc.dt.year.astype(str) + "_" + df.dtc.dt.month.map(
    {12:"W",1:"W",2:"W",3:"Sp",4:"Sp",5:"Sp",6:"Su",7:"Su",8:"Su",9:"A",10:"A",11:"A"})
TRAITS=["milk_kg","fat_p","protein_p","SCS","ECM"]
CLIM=["THI_1_avg","temperature_avg","humidity_avg"]
rule("LOADED"); print(f"  rows={len(df):,} | summer(warmhalf)={df.is_summer_warmhalf.sum():,} "
                      f"| summer(strict)={df.is_summer_strict.sum():,}")

# %% [markdown]
# ## 3.1 Season-stratified correlations (the season confound)
# Spearman of each trait vs each climate driver: full-year vs summer vs winter. If the annual r
# is near zero but summer and winter have opposite signs, the annual value is a cancellation
# artefact (R2). Also Piscopo subsets (THI>72, ET>27°C).

# %%
rule("3.1  SEASON-STRATIFIED CORRELATIONS (Spearman)")
def scorr(sub, a, b):
    s=sub[[a,b]].dropna();
    return np.nan if len(s)<50 else s[a].corr(s[b], method="spearman")
rows=[]
sm = df[df.is_summer_warmhalf]; wn = df[~df.is_summer_warmhalf]
for t in TRAITS:
    for c in CLIM:
        rows.append({"trait":t,"climate":c,
                     "r_full":scorr(df,t,c),"r_summer":scorr(sm,t,c),"r_winter":scorr(wn,t,c),
                     "r_THI>72":scorr(df[df.THI_1_avg>72],t,c),
                     "r_ET>27":scorr(df[df.temperature_avg>27],t,c)})
cor=pd.DataFrame(rows).round(3); cor.to_csv(TAB/"season_stratified_corr.csv",index=False)
print(cor.to_string(index=False))
flip = cor[(np.sign(cor.r_summer)!=np.sign(cor.r_winter)) & cor.r_summer.notna()]
print(f"\n  sign-flip (summer vs winter) in {len(flip)}/{len(cor)} trait-climate pairs -> season confound present")

# %% [markdown]
# ## 3.2 THI-class response curves — raw vs within-animal, find the apparent optimum

# %%
rule("3.2  THI RESPONSE CURVES + OPTIMUM")
df["THI_band"] = pd.cut(df.THI_1_avg, bins=np.arange(45,90,3))
df["mc"] = df.groupby("Animal_ID")["milk_kg"].transform("mean")      # within-animal baselines
df["sc"] = df.groupby("Animal_ID")["SCS"].transform("mean")
g = df.groupby("THI_band", observed=True)
curve = pd.DataFrame({
    "n":g.size(),
    "milk_raw":g.milk_kg.mean(), "SCS_raw":g.SCS.mean(),
    "milk_wi":(df.milk_kg-df.mc).groupby(df.THI_band, observed=True).mean(),   # within-animal deviation
    "SCS_wi":(df.SCS-df.sc).groupby(df.THI_band, observed=True).mean(),
}).reset_index()
curve["THI_mid"]=curve.THI_band.apply(lambda b: b.mid)
curve=curve[curve.n>2000]                                            # drop sparse tails
curve.to_csv(TAB/"thi_response_curve.csv",index=False)
print(curve[["THI_band","n","milk_raw","milk_wi","SCS_raw","SCS_wi"]].round(3).to_string(index=False))
opt_milk_raw=curve.loc[curve.milk_raw.idxmax(),"THI_mid"]
opt_milk_wi =curve.loc[curve.milk_wi.idxmax(),"THI_mid"]
print(f"\n  milk optimum THI: raw={opt_milk_raw:.0f}, within-animal={opt_milk_wi:.0f}  (Matera buffalo optimum 59-63)")

fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ax[0].plot(curve.THI_mid,curve.milk_wi,marker="o"); ax[0].axhline(0,c="grey",lw=.8)
ax[0].set_title("Milk (within-animal deviation) vs THI"); ax[0].set_xlabel("THI_1_avg"); ax[0].set_ylabel("Δ milk_kg"); ax[0].grid(alpha=.3)
ax[1].plot(curve.THI_mid,curve.SCS_wi,marker="o",color="firebrick"); ax[1].axhline(0,c="grey",lw=.8)
ax[1].set_title("SCS (within-animal deviation) vs THI"); ax[1].set_xlabel("THI_1_avg"); ax[1].set_ylabel("Δ SCS"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG/"thi_response_curves.png",dpi=130); plt.show()
print(f"  wrote {FIG/'thi_response_curves.png'}")

# %% [markdown]
# ## 3.3 Positive control — Matera 2022 model (fixed effects + cluster-robust SE)
# Matera used a random animal effect; for the fixed-effect inference (is THI×parity significant?
# what is the optimum?) an OLS with the same fixed effects and farm-clustered SE is an equivalent,
# tractable stand-in on our 316-farm data. Fitted on a random sample for speed; LR covariate as
# in Matera (SCS on MY for FP/PP; MY on SCS; none for MY).

# %%
rule("3.3  MATERA POSITIVE CONTROL")
samp = df.sample(n=min(300_000, len(df)), random_state=42).copy()
samp["THI_band6"]=pd.cut(samp.THI_1_avg, bins=[0,58,62,66,70,74,200],
                         labels=["<58","58-62","62-66","66-70","70-74",">74"])
samp["pc"]=samp.parity_class.astype("category"); samp["d30"]=samp.dim30.astype("category")
LRcov={"milk_kg":None,"fat_p":"SCS","protein_p":"SCS","SCS":"milk_kg","ECM":None}
mat=[]
for t in TRAITS:
    terms="C(ys_calv) + C(THI_band6)*C(pc) + C(d30)*C(pc)"
    if LRcov[t]: terms+=f" + {LRcov[t]}"
    m=smf.ols(f"{t} ~ "+terms, data=samp).fit(cov_type="cluster", cov_kwds={"groups":samp.Farm_Code})
    wt=m.wald_test_terms(scalar=True)
    def pget(key):
        for idx in wt.table.index:
            if key in idx: return wt.table.loc[idx,"pvalue"]
        return np.nan
    p_thi=pget("C(THI_band6)") ; p_int=pget("C(THI_band6):C(pc)")
    mat.append({"trait":t,"p_THI":p_thi,"p_THIxParity":p_int})
    print(f"  {t:10s}  p(THI)={p_thi:.2e}   p(THI x parity)={p_int:.2e}")
matdf=pd.DataFrame(mat); matdf.to_csv(TAB/"matera_model_tests.csv",index=False)

# %% [markdown]
# ## 3.4 Causal within-animal anomaly model (de-confounded, summer)
# Within-animal fixed effects via demeaning; heat = THI anomaly (R1); control DIM, DIM², parity
# (R4); farm-clustered SE (R5). Run on both season definitions (D2). Also a full-year RAW-THI
# version to expose the season confound (wrong sign).

# %%
rule("3.4  WITHIN-ANIMAL ANOMALY MODEL")
def within_fe(data, ycol, xcol, clabel):
    d=data[[ycol,xcol,"DIM","DIM2","parity","Animal_ID","Farm_Code"]].dropna().copy()
    grp=d.groupby("Animal_ID")
    for col in [ycol,xcol,"DIM","DIM2","parity"]:
        d[col+"_w"]=d[col]-grp[col].transform("mean")
    m=smf.ols(f"{ycol}_w ~ {xcol}_w + DIM_w + DIM2_w + parity_w - 1",
              data=d).fit(cov_type="cluster", cov_kwds={"groups":d.Farm_Code})
    b=m.params[f"{xcol}_w"]; p=m.pvalues[f"{xcol}_w"]
    print(f"    {clabel:34s} {ycol:9s}  Δ per +10 {xcol}: {10*b:+.4f}  (p={p:.1e}, n={len(d):,})")
    return {"model":clabel,"trait":ycol,"x":xcol,"per10":round(10*b,4),"p":float(p),"n":int(len(d))}

res=[]
print("  -- causal: THI ANOMALY, summer=warm-half (Apr-Sep) --")
for t in ["milk_kg","SCS","fat_p","protein_p"]:
    res.append(within_fe(df[df.is_summer_warmhalf], t, "THI_anom", "anom / summer(warmhalf)"))
print("  -- robustness: THI ANOMALY, summer=strict (Jun-Sep) --")
for t in ["milk_kg","SCS"]:
    res.append(within_fe(df[df.is_summer_strict], t, "THI_anom", "anom / summer(strict)"))
print("  -- confound demo: RAW THI, full year (expect WRONG sign) --")
for t in ["milk_kg","SCS"]:
    res.append(within_fe(df, t, "THI_1_avg", "raw THI / full year"))
pd.DataFrame(res).to_csv(TAB/"within_animal_anomaly.csv",index=False)

# %% [markdown]
# ## 3.5 Non-linear shape: quadratic THI + SCS heat-threshold breakpoint

# %%
rule("3.5  NON-LINEAR SHAPE")
sm=df[df.is_summer_warmhalf][["milk_kg","SCS","THI_anom","DIM","DIM2","Animal_ID","Farm_Code"]].dropna().copy()
g=sm.groupby("Animal_ID")
for c in ["milk_kg","SCS","THI_anom","DIM","DIM2"]:
    sm[c+"_w"]=sm[c]-g[c].transform("mean")
sm["THI_anom_w2"]=sm.THI_anom_w**2
mq=smf.ols("milk_kg_w ~ THI_anom_w + THI_anom_w2 + DIM_w + DIM2_w - 1",data=sm).fit(
        cov_type="cluster",cov_kwds={"groups":sm.Farm_Code})
print(f"  milk quadratic THI_anom²: coef={mq.params['THI_anom_w2']:+.5f} (p={mq.pvalues['THI_anom_w2']:.1e}) "
      f"-> {'curvature present' if mq.pvalues['THI_anom_w2']<0.05 else '~linear'}")
# SCS heat threshold: breakpoint on the raw SCS-vs-THI curve (grid search, piecewise linear)
cv=curve.dropna(subset=["SCS_raw"])
x=cv.THI_mid.values.astype(float); y=cv.SCS_raw.values
best=None
for bp in np.arange(x.min()+3, x.max()-3, 1):
    X=np.column_stack([np.ones_like(x), x, np.maximum(x-bp,0)])
    beta,rss,*_=np.linalg.lstsq(X,y,rcond=None)
    rss=float(((y-X@beta)**2).sum())
    if best is None or rss<best[1]: best=(bp,rss,beta)
print(f"  SCS-vs-THI breakpoint ≈ THI {best[0]:.0f}  (slope change {best[2][2]:+.4f} SCS/THI above it)")

# %% [markdown]
# ## 3.6 Literature-comparison checkpoint

# %%
rule("3.6  LITERATURE COMPARISON")
milk_anom=[r for r in res if r["model"].startswith("anom / summer(warm") and r["trait"]=="milk_kg"][0]["per10"]
scs_anom =[r for r in res if r["model"].startswith("anom / summer(warm") and r["trait"]=="SCS"][0]["per10"]
milk_raw =[r for r in res if r["model"].startswith("raw") and r["trait"]=="milk_kg"][0]
verdict("Season confound: annual r cancels (summer/winter opposite)", len(flip)>0, f"{len(flip)} pairs flip")
verdict("Causal: heat (anomaly) LOWERS milk in summer", milk_anom<0, f"{milk_anom:+.3f} kg / +10 THI-anom")
verdict("Causal: heat (anomaly) RAISES SCS in summer (Matera)", scs_anom>0, f"{scs_anom:+.3f} SCS / +10 THI-anom")
verdict("Confound: raw THI gives WRONG (beneficial) milk sign", milk_raw["per10"]>0,
        f"raw THI milk {milk_raw['per10']:+.3f} (positive = artefact)")
verdict("Matera: THI×parity significant for PP & SCS",
        (matdf.set_index('trait').loc['protein_p','p_THIxParity']<0.05) and
        (matdf.set_index('trait').loc['SCS','p_THIxParity']<0.05),
        f"p(PP)={matdf.set_index('trait').loc['protein_p','p_THIxParity']:.1e}, "
        f"p(SCS)={matdf.set_index('trait').loc['SCS','p_THIxParity']:.1e}")
ecm_summer=scorr(sm if False else df[df.is_summer_warmhalf],"ECM","temperature_avg")
verdict("Piscopo: ECM negatively correlated with heat in summer", ecm_summer<0, f"Spearman(ECM,temp,summer)={ecm_summer:+.3f}")

summary={"n_sign_flips":int(len(flip)),"milk_opt_THI_raw":float(opt_milk_raw),"milk_opt_THI_within":float(opt_milk_wi),
         "milk_per10_anom_summer":milk_anom,"SCS_per10_anom_summer":scs_anom,
         "milk_per10_rawTHI_fullyear":milk_raw["per10"],"SCS_breakpoint_THI":float(best[0])}
(TAB/"stage3_summary.json").write_text(json.dumps(summary,indent=2))
rule("STAGE 3 COMPLETE"); print("  summary:",summary)

# %% [markdown]
# ## 3.7 Figures — climate signal

# %%
rule("3.7  FIGURES")
# Fig 3.1: season-stratified correlations (THI vs each trait): full vs summer vs winter
ct=cor[cor.climate=="THI_1_avg"].set_index("trait")[["r_full","r_summer","r_winter"]]
fig,ax=plt.subplots(figsize=(9,4.5)); x=np.arange(len(ct)); w=0.26
for i,(cname,col) in enumerate(zip(["r_full","r_summer","r_winter"],["grey","firebrick","steelblue"])):
    ax.bar(x+(i-1)*w,ct[cname],w,label=cname,color=col)
ax.axhline(0,c="k",lw=.8); ax.set_xticks(x); ax.set_xticklabels(ct.index)
ax.set_title("3.1  Spearman(trait, THI): annual ≈ 0 hides summer/winter cancellation"); ax.set_ylabel("Spearman r"); ax.legend(); ax.grid(alpha=.3,axis="y")
plt.tight_layout(); plt.savefig(FIG/"s3_season_corr.png",dpi=130); plt.show()

# Fig 3.3: Matera positive control — significance of THI and THI×parity
mm=matdf.set_index("trait")
fig,ax=plt.subplots(figsize=(9,4.5)); x=np.arange(len(mm)); w=0.38
ax.bar(x-w/2,-np.log10(mm.p_THI),w,label="THI main effect",color="darkorange")
ax.bar(x+w/2,-np.log10(mm.p_THIxParity),w,label="THI × parity",color="purple")
ax.axhline(-np.log10(0.05),c="red",ls="--",label="p=0.05")
ax.set_xticks(x); ax.set_xticklabels(mm.index); ax.set_ylabel("-log10(p)")
ax.set_title("3.3  Matera control: THI significant for all traits; THI×parity for milk/fat/SCS"); ax.legend(); ax.grid(alpha=.3,axis="y")
plt.tight_layout(); plt.savefig(FIG/"s3_matera_significance.png",dpi=130); plt.show()

# Fig 3.4: causal within-animal effects per +10 (anomaly vs raw, milk vs SCS)
rr=pd.DataFrame(res)
fig,ax=plt.subplots(figsize=(10,4.5))
lab=[f"{r.trait}\n{r.model}" for r in rr.itertuples()]
colors=["firebrick" if v>0 else "steelblue" for v in rr.per10]
ax.barh(range(len(rr)),rr.per10,color=colors); ax.axvline(0,c="k",lw=.8)
ax.set_yticks(range(len(rr))); ax.set_yticklabels(lab,fontsize=7)
ax.set_xlabel("Δ per +10 THI (units of trait)")
ax.set_title("3.4  Within-animal heat effect: anomaly (correct) vs raw THI (confounded, wrong sign)")
ax.grid(alpha=.3,axis="x"); plt.tight_layout(); plt.savefig(FIG/"s3_within_anomaly_effects.png",dpi=130); plt.show()

# Fig 3.5: SCS-vs-THI breakpoint fit
cv=curve.dropna(subset=["SCS_raw"]); xv=cv.THI_mid.values.astype(float); yv=cv.SCS_raw.values
bp=best[0]; be=best[2]; xg=np.linspace(xv.min(),xv.max(),100)
yhat=be[0]+be[1]*xg+be[2]*np.maximum(xg-bp,0)
plt.figure(figsize=(7,4.5))
plt.scatter(xv,yv,s=25,color="firebrick",label="SCS mean per THI band")
plt.plot(xg,yhat,color="k",label=f"segmented fit (bp≈{bp:.0f})"); plt.axvline(bp,c="grey",ls="--")
plt.xlabel("THI_1_avg"); plt.ylabel("SCS (raw mean)"); plt.title("3.5  SCS–THI breakpoint (raw, confounded)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG/"s3_scs_breakpoint.png",dpi=130); plt.show()
print("  wrote s3_season_corr / s3_matera_significance / s3_within_anomaly_effects / s3_scs_breakpoint")
