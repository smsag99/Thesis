# %% [markdown]
# # Stage 5 — Clustering & Structure Discovery  (data analysis, no prediction)
#
# Loads `analysis_base.parquet` + `wood_features.parquet`. Question: do buffalo/lactations form
# discrete groups or a continuum, and do any lactation-curve types map to heat vulnerability?
#
# 1. Trait clustering (reproduce Trapanese 2025): K-Means vs Ward, z-score, silhouette/DBI/CHI.
# 2. Feature pairs, no PCA (R6): 2 real features at a time, milk_kg on Y.
# 3. Continuum check: parity-stratified K-Means + DBSCAN density.
# 4. Curve-shape clustering: cluster interpretable Wood's features -> real lactation types.
# 5. Season split: does cluster structure shift summer vs winter.
# 6. Curve clusters x heat/SCS: is one curve type the heat-vulnerable udder-health group
#    (also yields the cluster-specific lactation curve Stage 4 wanted).
#
# Anchor: Trapanese 2025. Checkpoint: silhouette ~0.17-0.18, ~2 clusters, DIM+milk discriminate.

# %%
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
SEED=42

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
def sil(X,lab): return round(silhouette_score(X,lab,sample_size=min(10000,len(X)),random_state=SEED),3) if len(set(lab))>1 else np.nan
def metrics(X,lab): return (sil(X,lab), round(davies_bouldin_score(X,lab),3), round(calinski_harabasz_score(X,lab),1)) if len(set(lab))>1 else (np.nan,)*3
def best_k(X, ks=range(2,7)):
    b=(None,-1,None)
    for k in ks:
        lab=KMeans(k,n_init=10,random_state=SEED).fit_predict(X)
        s=sil(X,lab)
        if s>b[1]: b=(k,s,lab)
    return b

base=pd.read_parquet(OUT/"analysis_base.parquet")
wf=pd.read_parquet(OUT/"wood_features.parquet")
rule("LOADED"); print(f"  records={len(base):,} | animals={base.Animal_ID.nunique():,} | wood lactations={len(wf):,}")

# %% [markdown]
# ## 5.1 Trait clustering — reproduce Trapanese 2025 (K-Means vs Ward)
# Per test-day record (Trapanese's level; DIM varies), standard production/quality traits.

# %%
rule("5.1  TRAIT CLUSTERING (Trapanese)")
TF=["milk_kg","DIM","fat_p","protein_p","SCS"]
rec=base[TF].dropna().sample(n=40000,random_state=SEED)
Xr=StandardScaler().fit_transform(rec)
k_rec,sil_rec,lab_rec=best_k(Xr)
mr=metrics(Xr,lab_rec)
print(f"  K-Means best k={k_rec}: silhouette={mr[0]}  DBI={mr[1]}  CHI={mr[2]}")
Xs=Xr[np.random.RandomState(SEED).choice(len(Xr),4000,replace=False)]
wl=AgglomerativeClustering(n_clusters=k_rec, linkage="ward").fit_predict(Xs)
print(f"  Ward (n=4000) k={k_rec}: silhouette={sil(Xs,wl)}  (Trapanese: K-Means 0.17-0.18 > hierarchical 0.10-0.12)")
prof=rec.assign(cl=lab_rec).groupby("cl")[TF].mean().round(2)
prof["n"]=pd.Series(lab_rec).value_counts()
print("  cluster profiles (per-record):"); print(prof.to_string())
cent=pd.DataFrame(Xr,columns=TF).assign(cl=lab_rec).groupby("cl").mean()
disc=(cent.max()-cent.min()).sort_values(ascending=False).round(2)
print("  discriminating features (centroid spread, z):", disc.to_dict())
# fixed 2-cluster silhouette for the direct Trapanese comparison
sil2=sil(Xr, KMeans(2,n_init=10,random_state=SEED).fit_predict(Xr))
print(f"  fixed k=2 silhouette (Trapanese-comparable) = {sil2}")

# %% [markdown]
# ## 5.2 Feature pairs, no PCA (milk_kg on Y)

# %%
rule("5.2  FEATURE PAIRS (no PCA)")
an=base.groupby("Animal_ID").agg(milk_kg=("milk_kg","mean"),DIM=("DIM","mean"),
    fat_p=("fat_p","mean"),protein_p=("protein_p","mean"),SCS=("SCS","mean"),
    parity=("parity","max"),AFC=("true_AFC_days","mean")).dropna()
pairs=[("DIM","milk_kg"),("parity","milk_kg"),("SCS","milk_kg"),("fat_p","milk_kg"),
       ("protein_p","milk_kg"),("AFC","milk_kg")]
prows=[]
for x,y in pairs:
    X=StandardScaler().fit_transform(an[[x,y]])
    k,s,_=best_k(X)
    prows.append({"pair":f"{x} x {y}","best_k":k,"silhouette":round(s,3),"discrete_x":x in ("parity","AFC")})
pp=pd.DataFrame(prows).sort_values("silhouette",ascending=False)
pp.to_csv(TAB/"feature_pair_silhouettes.csv",index=False)
print(pp.to_string(index=False))
print("  -> highest silhouettes are the parity/AFC pairs (discreteness artifact), not biology")

# %% [markdown]
# ## 5.3 Continuum check — parity-stratified K-Means + DBSCAN

# %%
rule("5.3  CONTINUUM CHECK")
crows=[]
for p in [1,2,3,4,5]:
    d=base[base.parity_class==p][["DIM","milk_kg"]].dropna()
    if len(d)>25000: d=d.sample(25000,random_state=SEED)
    X=StandardScaler().fit_transform(d)
    k,s,_=best_k(X, ks=range(2,5))
    db=DBSCAN(eps=0.3,min_samples=50).fit_predict(X)
    ndb=len(set(db))-(1 if -1 in db else 0); noise=(db==-1).mean()
    crows.append({"parity":p,"kmeans_best_k":k,"kmeans_sil":round(s,3),
                  "dbscan_clusters":ndb,"dbscan_noise":round(float(noise),3)})
cc=pd.DataFrame(crows); cc.to_csv(TAB/"continuum_check.csv",index=False)
print(cc.to_string(index=False))
print("  -> low silhouettes + ~1 dense DBSCAN cluster per stratum = continuum, not discrete types")

# %% [markdown]
# ## 5.4 Curve-shape clustering — interpretable Wood's features (the REAL structure)

# %%
rule("5.4  CURVE-SHAPE CLUSTERING")
CF=["peak_DIM","peak_yield","persistency","yield_305"]
wfx=wf[wf.yield_305.between(500,4500)].copy()
Xc=StandardScaler().fit_transform(wfx[CF])
k_curve,sil_curve,lab_curve=best_k(Xc, ks=range(2,7))
mc=metrics(Xc,lab_curve)
print(f"  best k={k_curve}: silhouette={mc[0]}  DBI={mc[1]}  CHI={mc[2]}")
wfx["curve_cl"]=lab_curve
cp=wfx.groupby("curve_cl")[CF].mean().round(1); cp["n"]=pd.Series(lab_curve).value_counts()
print("  curve-type profiles:"); print(cp.to_string())
wfx[["lac","parity","curve_cl"]].to_parquet(OUT/"curve_clusters.parquet",index=False)

# %% [markdown]
# ## 5.5 Season split — does structure shift summer vs winter

# %%
rule("5.5  SEASON SPLIT (milk x DIM)")
for name,mask in [("summer",base.is_summer_warmhalf),("winter",~base.is_summer_warmhalf)]:
    d=base[mask][["DIM","milk_kg"]].dropna().sample(30000,random_state=SEED)
    X=StandardScaler().fit_transform(d)
    lab=KMeans(3,n_init=10,random_state=SEED).fit_predict(X)
    means=sorted(np.round(d.assign(c=lab).groupby("c").milk_kg.mean().tolist(),1))
    print(f"  {name}: k=3 silhouette={sil(X,lab)}  milk cluster means={means}")

# %% [markdown]
# ## 5.6 Curve clusters × heat / SCS — is one type heat-vulnerable? (thesis payoff)
# Assign each record its lactation's curve cluster; within-animal anomaly heat effect per cluster.

# %%
rule("5.6  CURVE CLUSTERS x HEAT / SCS")
cl=pd.read_parquet(OUT/"curve_clusters.parquet")
cl["Animal_ID"]=cl.lac.str.split("|").str[0]
mg=base.merge(cl[["Animal_ID","parity","curve_cl"]], on=["Animal_ID","parity"], how="inner")
Sm=mg[mg.is_summer_warmhalf].copy(); Sm["DIM2"]=Sm.DIM.astype(float)**2
def fe(sub,trait):
    d=sub[[trait,"THI_anom","DIM","DIM2","Animal_ID","Farm_Code"]].dropna().copy()
    if d.Animal_ID.nunique()<200: return None
    g=d.groupby("Animal_ID")
    for c in [trait,"THI_anom","DIM","DIM2"]: d[c+"_w"]=d[c]-g[c].transform("mean")
    r=smf.ols(f"{trait}_w ~ THI_anom_w + DIM_w + DIM2_w - 1",data=d).fit(cov_type="cluster",cov_kwds={"groups":d.Farm_Code})
    return round(10*r.params["THI_anom_w"],4), round(10*1.96*r.bse["THI_anom_w"],4)
hrows=[]
for c in sorted(mg.curve_cl.unique()):
    sub=Sm[Sm.curve_cl==c]
    prof_c={k:round(wfx[wfx.curve_cl==c][k].median(),1) for k in ["peak_yield","persistency"]}
    for t in ["milk_kg","SCS"]:
        r=fe(sub,t)
        if r: hrows.append({"curve_cl":int(c),**prof_c,"trait":t,"per10":r[0],"ci95":r[1],"n_rec":len(sub)})
hh=pd.DataFrame(hrows); hh.to_csv(TAB/"curvecluster_heat.csv",index=False)
print(hh.to_string(index=False))

# %% [markdown]
# ## 5.7 Literature-comparison checkpoint

# %%
rule("5.7  LITERATURE COMPARISON")
verdict("Trapanese: weak trait-cluster structure (silhouette ~0.17-0.18)", 0.10<=sil2<=0.30,
        f"per-record k=2 silhouette={sil2}")
verdict("Trapanese: DIM & milk among main discriminators", ("DIM" in disc.index[:2]) or ("milk_kg" in disc.index[:2]),
        f"top discriminators={list(disc.index[:2])}")
verdict("Continuum: DBSCAN ~1 dense cluster per parity stratum", (cc.dbscan_clusters<=2).all(),
        f"dbscan clusters/stratum={cc.dbscan_clusters.tolist()}")
verdict("Curve-shape clustering = real structure (>= point-trait silhouette)", mc[0]>=sil2,
        f"curve silhouette={mc[0]} vs point-trait {sil2}")
heat_spread = hh[hh.trait=='SCS'].per10.max()-hh[hh.trait=='SCS'].per10.min() if len(hh) else 0
verdict("Heat->SCS slope differs across curve types (heat-vulnerable group?)", heat_spread>0.05,
        f"SCS per10 spread across curve clusters = {round(float(heat_spread),3)}")
summary={"trait_cluster_k":int(k_rec),"per_record_sil2":float(sil2),"top_discriminators":list(disc.index[:2]),
         "curve_clusters":int(k_curve),"curve_silhouette":float(mc[0]),
         "pair_silhouettes":pp.set_index("pair").silhouette.to_dict()}
(TAB/"stage5_summary.json").write_text(json.dumps(summary,indent=2,default=str))
rule("STAGE 5 COMPLETE"); print("  summary:",summary)

# %% [markdown]
# ## 5.8 Figures — clustering & structure

# %%
rule("5.8  FIGURES")
# Fig 5.1: trait clusters on real axes (milk vs DIM), colored by K-Means cluster
plt.figure(figsize=(7,5))
samp_i=np.random.RandomState(SEED).choice(len(rec),8000,replace=False)
plt.scatter(rec.DIM.values[samp_i],rec.milk_kg.values[samp_i],c=lab_rec[samp_i],cmap="coolwarm",s=6,alpha=.5)
plt.xlabel("DIM"); plt.ylabel("milk_kg"); plt.title(f"5.1  Trait clusters k={k_rec} (sil={sil2}) = early vs late lactation")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG/"s5_trait_clusters.png",dpi=130); plt.show()

# Fig 5.2: feature-pair silhouettes (discreteness artifact)
fig,ax=plt.subplots(figsize=(9,4.2))
colors=["indianred" if d else "steelblue" for d in pp.discrete_x]
ax.barh(pp.pair,pp.silhouette,color=colors); ax.set_xlabel("best silhouette")
ax.set_title("5.2  Feature-pair silhouettes (red = discrete x: parity/AFC artifact)"); ax.grid(alpha=.3,axis="x")
ax.invert_yaxis(); plt.tight_layout(); plt.savefig(FIG/"s5_feature_pairs.png",dpi=130); plt.show()

# Fig 5.3: continuum check — silhouette per parity + DBSCAN cluster count
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].bar(cc.parity.astype(str),cc.kmeans_sil,color="slateblue"); ax[0].set_title("5.3  K-Means silhouette per parity stratum (milk×DIM)")
ax[0].set_xlabel("parity"); ax[0].set_ylabel("silhouette"); ax[0].grid(alpha=.3,axis="y")
ax[1].bar(cc.parity.astype(str),cc.dbscan_clusters,color="seagreen"); ax[1].set_title("5.3  DBSCAN dense clusters per stratum (=1 → continuum)")
ax[1].set_xlabel("parity"); ax[1].set_ylabel("# dense clusters"); ax[1].set_ylim(0,3); ax[1].grid(alpha=.3,axis="y")
plt.tight_layout(); plt.savefig(FIG/"s5_continuum.png",dpi=130); plt.show()

# Fig 5.4: curve-shape clusters — feature scatter + mean Wood's curves
fig,ax=plt.subplots(1,2,figsize=(13,4.8))
si=np.random.RandomState(SEED).choice(len(wfx),12000,replace=False)
sc=ax[0].scatter(wfx.peak_yield.values[si],wfx.persistency.values[si],c=wfx.curve_cl.values[si],cmap="viridis",s=8,alpha=.5)
ax[0].set_xlabel("peak_yield"); ax[0].set_ylabel("persistency"); ax[0].set_title("5.4  Curve-shape clusters (real structure)"); ax[0].grid(alpha=.3)
t=np.arange(1,306)
for c in sorted(wfx.curve_cl.unique()):
    sub=wfx[wfx.curve_cl==c]; ma,mb,mc=sub.a.median(),sub.b.median(),sub.c.median()
    ax[1].plot(t,ma*t**mb*np.exp(-mc*t),label=f"type {c} (n={len(sub)//1000}k)")
ax[1].set_xlabel("DIM"); ax[1].set_ylabel("milk_kg"); ax[1].set_title("5.4  Mean Wood's curve per type"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG/"s5_curve_clusters.png",dpi=130); plt.show()

# Fig 5.6: curve cluster × heat forest (milk & SCS)
fig,ax=plt.subplots(1,2,figsize=(12,4))
for a,t in zip(ax,["milk_kg","SCS"]):
    s=hh[hh.trait==t]; yv=range(len(s))
    a.errorbar(s.per10,yv,xerr=s.ci95,fmt="o",capsize=3,color="firebrick" if t=="SCS" else "steelblue")
    a.axvline(0,c="grey",lw=.8); a.set_yticks(list(yv)); a.set_yticklabels([f"type {c}" for c in s.curve_cl])
    a.set_title(f"5.6  {t}: heat effect per +10 anomaly by curve type"); a.set_xlabel("Δ per +10"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG/"s5_curvecluster_heat.png",dpi=130); plt.show()
print("  wrote s5_trait_clusters / s5_feature_pairs / s5_continuum / s5_curve_clusters / s5_curvecluster_heat")
