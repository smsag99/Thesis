# %% [markdown]
# # Stage 2 — Descriptive & Phenotypic Characterization
#
# Characterize the population and the traits (the "Table 1 / Figure 1" of the thesis).
# Loads `outputs/analysis_base.parquet` from Stage 1. ANALYSIS-ONLY (no prediction).
#
# 1. Descriptive statistics by parity and stage of lactation (Matera Table 1 / Costa tables).
# 2. Lactation curves: Wood's function fit per lactation -> peak, persistency, 305-d yield.
# 3. Trait correlations (Pearson + Spearman).
# 4. Variance decomposition / repeatability (Costa 2020 mixed-model spirit).
# 5. Literature-comparison checkpoint (3rd parity most productive; PP nadir ~80 DIM; SCS rises with DIM).
#
# Anchors: Costa et al. 2020 (phenotypic characterization, PROC MIXED), Matera et al. 2022 (Table 1).

# %%
import pandas as pd, numpy as np, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gamma as gammafn, gammainc
import statsmodels.formula.api as smf

RNG = np.random.default_rng(42)

def find_root():
    cands = [Path("Thesis_Data"), Path("../../Thesis_Data")]
    try: cands.append(Path(__file__).resolve().parents[2] / "Thesis_Data")
    except NameError: pass
    for c in cands:
        if c.exists(): return c.resolve().parent
    raise SystemExit("Cannot locate Thesis_Data")
ROOT = find_root()
OUT = ROOT / "codes" / "pipeline" / "outputs"
FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = OUT / "tables";  TAB.mkdir(parents=True, exist_ok=True)

def rule(m): print("\n"+"="*78+f"\n{m}\n"+"="*78)
def verdict(name, ok, detail=""):
    print(f"  [{'MATCH' if ok else 'CHECK'}] {name}" + (f"  --  {detail}" if detail else ""))

df = pd.read_parquet(OUT / "analysis_base.parquet")
df["HTD"] = df.Farm_Code.astype(str) + "_" + df.dtt.astype(str)   # herd-test-date contemporary group
TRAITS = ["milk_kg","fat_p","protein_p","SCS","ECM"]
rule("LOADED")
print(f"  rows={len(df):,} | animals={df.Animal_ID.nunique():,} | farms={df.Farm_Code.nunique()} "
      f"| lactations={df.groupby(['Animal_ID','parity']).ngroups:,}")

# %% [markdown]
# ## 2.1 Descriptive statistics by parity and by stage of lactation

# %%
rule("2.1  DESCRIPTIVE STATISTICS")
# By parity class (Matera Table 1 layout: mean +/- SD per trait)
by_par = df.groupby("parity_class")[TRAITS].agg(["mean","std"]).round(3)
by_par["n"] = df.groupby("parity_class").size()
print("\n-- by parity class (1..5+) --")
print(by_par.to_string())
by_par.to_csv(TAB / "desc_by_parity.csv")

# By stage of lactation (30-day classes)
by_sol = df.groupby("dim30")[TRAITS].mean().round(3)
by_sol["n"] = df.groupby("dim30").size()
by_sol.to_csv(TAB / "desc_by_SOL.csv")
print("\n-- trait means by SOL class (dim30) --")
print(by_sol.to_string())

# overall
overall = df[TRAITS].describe(percentiles=[.05,.5,.95]).T[["mean","std","5%","50%","95%"]].round(3)
overall.to_csv(TAB / "desc_overall.csv")

# %% [markdown]
# ## 2.2 Lactation curves — Wood's function y(t)=a·t^b·e^(−ct)
# Fitted per lactation by log-linearisation: ln y = ln a + b·ln t − c·t (OLS via per-group
# sufficient statistics, fully vectorised). Features: peak DIM=b/c, peak yield, Wood's
# persistency=−(b+1)ln c, and 305-d yield via the lower incomplete gamma integral.

# %%
rule("2.2  LACTATION CURVES (WOOD'S)")
w = df[["Animal_ID","parity","DIM","milk_kg"]].copy()
w = w[(w.DIM > 0) & (w.milk_kg > 0)]
w["x1"] = np.log(w.DIM)          # ln t
w["x2"] = w.DIM.astype(float)    # t
w["y"]  = np.log(w.milk_kg)      # ln milk
w["lac"] = w.Animal_ID + "|" + w.parity.astype(str)
# per-lactation sufficient statistics for OLS of y on [1, x1, x2]
w["x1x1"]=w.x1*w.x1; w["x2x2"]=w.x2*w.x2; w["x1x2"]=w.x1*w.x2
w["x1y"]=w.x1*w.y;   w["x2y"]=w.x2*w.y
agg = w.groupby("lac").agg(
    n=("y","size"), Sx1=("x1","sum"), Sx2=("x2","sum"),
    Sx1x1=("x1x1","sum"), Sx2x2=("x2x2","sum"), Sx1x2=("x1x2","sum"),
    Sy=("y","sum"), Sx1y=("x1y","sum"), Sx2y=("x2y","sum"),
    dmin=("x2","min"), dmax=("x2","max"))
# keep well-spanned lactations so the shape is real, not extrapolation
ok = (agg.n >= 5) & (agg.dmin < 60) & (agg.dmax > 200)
a = agg[ok]
N = len(a)
A = np.empty((N,3,3)); rhs = np.empty((N,3))
A[:,0,0]=a.n;    A[:,0,1]=a.Sx1;  A[:,0,2]=a.Sx2
A[:,1,0]=a.Sx1;  A[:,1,1]=a.Sx1x1;A[:,1,2]=a.Sx1x2
A[:,2,0]=a.Sx2;  A[:,2,1]=a.Sx1x2;A[:,2,2]=a.Sx2x2
rhs[:,0]=a.Sy;   rhs[:,1]=a.Sx1y; rhs[:,2]=a.Sx2y
beta = np.full((N,3), np.nan)
good = np.abs(np.linalg.det(A)) > 1e-8            # skip singular (degenerate) lactations
beta[good] = np.linalg.solve(A[good], rhs[good, :, None])[:, :, 0]
res = pd.DataFrame({"lac": a.index, "n": a.n.values,
                    "a": np.exp(beta[:,0]), "b": beta[:,1], "c": -beta[:,2]})
res = res[(res.b > 0) & (res.c > 0)]                      # biologically valid Wood's shape
res["peak_DIM"]   = res.b / res.c
res["peak_yield"] = res.a * (res.b/res.c)**res.b * np.exp(-res.b)
res["persistency"]= -(res.b + 1) * np.log(res.c)
res["yield_305"]  = res.a * gammafn(res.b+1) / res.c**(res.b+1) * gammainc(res.b+1, res.c*305)
res = res[(res.peak_DIM.between(1,305)) & (res.peak_yield.between(0.4,40))]  # drop degenerate fits
res[["parity"]] = res.lac.str.split("|", expand=True)[[1]].astype(int)
print(f"  lactations fitted (valid Wood's): {len(res):,} of {agg.shape[0]:,} candidate lactations")
print("  median curve features:", {k: round(float(res[k].median()),2)
      for k in ["peak_DIM","peak_yield","persistency","yield_305"]})
print("\n  median features by parity class:")
res["parity_class"] = np.minimum(res.parity, 5)
print(res.groupby("parity_class")[["peak_DIM","peak_yield","persistency","yield_305"]].median().round(2).to_string())
res.to_parquet(OUT / "wood_features.parquet", index=False)   # reused in Stage 5

# Figure: observed average lactation curve by parity (20-day DIM bins)
df["dimbin"] = (df.DIM//20)*20 + 10
curve = df[df.parity_class<=5].groupby(["parity_class","dimbin"])["milk_kg"].mean().reset_index()
plt.figure(figsize=(8,5))
for p in sorted(curve.parity_class.unique()):
    s = curve[curve.parity_class==p]
    plt.plot(s.dimbin, s.milk_kg, marker="", label=f"parity {int(p)}"+("+" if p==5 else ""))
plt.xlabel("DIM (days)"); plt.ylabel("mean milk_kg"); plt.title("Average lactation curve by parity")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(FIG / "lactation_curve_by_parity.png", dpi=130); plt.show()
print(f"  wrote {FIG/'lactation_curve_by_parity.png'}")

# %% [markdown]
# ## 2.3 Trait correlations (Pearson + Spearman)

# %%
rule("2.3  CORRELATIONS")
cols = ["milk_kg","fat_p","protein_p","SCS","ECM","DIM","parity","true_AFC_days"]
cc = df[cols].dropna()
pear = cc.corr(method="pearson").round(3)
spear = cc.corr(method="spearman").round(3)
pear.to_csv(TAB / "corr_pearson.csv"); spear.to_csv(TAB / "corr_spearman.csv")
print("\n-- Pearson (milk_kg, SCS rows) --")
print(pear.loc[["milk_kg","SCS"]].to_string())
print("\n-- Spearman (milk_kg, SCS rows) --")
print(spear.loc[["milk_kg","SCS"]].to_string())
# heatmap
plt.figure(figsize=(7,6))
im = plt.imshow(spear, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xticks(range(len(cols)), cols, rotation=45, ha="right"); plt.yticks(range(len(cols)), cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        plt.text(j,i,f"{spear.iloc[i,j]:.2f}",ha="center",va="center",fontsize=7)
plt.colorbar(im, shrink=.8); plt.title("Spearman correlation"); plt.tight_layout()
plt.savefig(FIG / "corr_spearman_heatmap.png", dpi=130); plt.show()
print(f"  wrote {FIG/'corr_spearman_heatmap.png'}")

# %% [markdown]
# ## 2.4 Variance decomposition / repeatability (Costa 2020 spirit)
# Full crossed animal×HTD REML on 1.6M rows is impractical in Python (Costa used SAS). We
# estimate, per trait, on a random subsample: (i) animal repeatability from a mixed model with
# fixed parity/SOL/season + random animal intercept; (ii) the herd-test-date variance share (eta²).

# %%
rule("2.4  VARIANCE DECOMPOSITION (sample-based)")
# sample whole animals to keep records grouped
an = df.Animal_ID.drop_duplicates().sample(n=min(8000, df.Animal_ID.nunique()), random_state=42)
s = df[df.Animal_ID.isin(set(an))].copy()
print(f"  sample: {len(s):,} records / {s.Animal_ID.nunique():,} animals")
rows = []
for t in TRAITS:
    d = s[[t,"parity_class","dim30","season4","Animal_ID","HTD"]].dropna().copy()
    d["dim30"]=d.dim30.astype("category"); d["parity_class"]=d.parity_class.astype("category")
    md = smf.mixedlm(f"{t} ~ C(parity_class) + C(dim30) + C(season4)", d, groups=d["Animal_ID"])
    mf = md.fit(method="lbfgs", reml=True)
    va = float(mf.cov_re.iloc[0,0]); ve = float(mf.scale)
    repeat = va/(va+ve)
    # HTD variance share = eta^2 of one-way grouping by contemporary group
    grp = d.groupby("HTD")[t]; gm = grp.transform("mean"); tot = d[t].var()
    eta_htd = ((gm - d[t].mean())**2).mean() / (d[t].var(ddof=0))
    rows.append({"trait":t, "animal_repeatability":round(repeat,3),
                 "HTD_var_share_eta2":round(float(eta_htd),3),
                 "sigma2_animal":round(va,3), "sigma2_resid":round(ve,3)})
vd = pd.DataFrame(rows); vd.to_csv(TAB / "variance_decomposition.csv", index=False)
print(vd.to_string(index=False))

# %% [markdown]
# ## 2.5 Literature-comparison checkpoint

# %%
rule("2.5  LITERATURE COMPARISON")
mpar = df.groupby("parity_class")["milk_kg"].mean()
best_par = int(mpar.idxmax())
verdict("Costa/Matera: 3rd parity most productive", best_par==3,
        f"peak at parity_class={best_par} ({mpar.max():.2f} kg); by parity: "
        + ", ".join(f"{int(p)}:{v:.2f}" for p,v in mpar.items()))
pp_by = df.groupby("dimbin")["protein_p"].mean(); pp_nadir=int(pp_by.idxmin())
verdict("Matera: protein_p nadir ~60-100 DIM", 40<=pp_nadir<=110, f"nadir at DIM~{pp_nadir}")
scs_dim = df[["SCS","DIM"]].corr(method="spearman").iloc[0,1]
verdict("Matera: SCS rises through lactation", scs_dim>0, f"Spearman(SCS,DIM)=+{scs_dim:.3f}")

summary = {"rows":int(len(df)), "lactations_fitted":int(len(res)),
           "best_parity_milk":best_par, "protein_nadir_DIM":pp_nadir,
           "SCS_DIM_spearman":round(float(scs_dim),3),
           "median_peak_DIM":round(float(res.peak_DIM.median()),1),
           "median_yield_305":round(float(res.yield_305.median()),1)}
(TAB / "stage2_summary.json").write_text(json.dumps(summary, indent=2))
rule("STAGE 2 COMPLETE")
print("  tables -> outputs/tables/ , figures -> outputs/figures/ , wood_features.parquet saved")
print("  summary:", summary)

# %% [markdown]
# ## 2.6 Figures — descriptive & phenotypic

# %%
rule("2.6  FIGURES")
# Fig 2.1a: trait means by parity class
pc=df.groupby("parity_class")[TRAITS].mean()
fig,ax=plt.subplots(2,3,figsize=(15,7))
for a,t in zip(ax.flat,TRAITS):
    pc[t].plot(kind="bar",ax=a,color="steelblue"); a.set_title(f"{t} by parity"); a.set_xlabel("parity class"); a.grid(alpha=.3,axis="y")
ax.flat[5].axis("off")
plt.suptitle("2.1  Trait means by parity class",y=1.02); plt.tight_layout(); plt.savefig(FIG/"s2_traits_by_parity.png",dpi=130); plt.show()

# Fig 2.1b: trait trends by stage of lactation
so=df.groupby("dim30")[TRAITS].mean()
fig,ax=plt.subplots(1,2,figsize=(13,4.5))
ax[0].plot(so.index,so.milk_kg,marker="o",label="milk_kg"); ax[0].plot(so.index,so.ECM,marker="s",label="ECM")
ax[0].set_title("Milk / ECM by SOL"); ax[0].set_xlabel("dim30 (30-d class)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(so.index,so.fat_p,marker="o",color="orange",label="fat%"); ax[1].plot(so.index,so.protein_p,marker="s",color="green",label="protein%")
axb=ax[1].twinx(); axb.plot(so.index,so.SCS,marker="^",color="firebrick",label="SCS")
ax[1].set_title("Composition & SCS by SOL"); ax[1].set_xlabel("dim30"); ax[1].legend(loc="upper left"); axb.legend(loc="upper right"); ax[1].grid(alpha=.3)
plt.suptitle("2.1  Trait trends across lactation",y=1.02); plt.tight_layout(); plt.savefig(FIG/"s2_traits_by_SOL.png",dpi=130); plt.show()

# Fig 2.2: Wood's curve feature distributions
fig,ax=plt.subplots(1,4,figsize=(16,3.7))
for a,c in zip(ax,["peak_DIM","peak_yield","persistency","yield_305"]):
    a.hist(res[c],bins=60,color="slateblue"); a.axvline(res[c].median(),color="k",ls="--",lw=1)
    a.set_title(f"{c} (med {res[c].median():.0f})"); a.grid(alpha=.3)
plt.suptitle("2.2  Wood's lactation-curve feature distributions",y=1.03); plt.tight_layout(); plt.savefig(FIG/"s2_wood_features_hist.png",dpi=130); plt.show()

# Fig 2.4: variance decomposition (two perspectives, grouped)
vv=vd.set_index("trait")
fig,ax=plt.subplots(figsize=(9,4.5)); x=np.arange(len(vv)); w=0.38
ax.bar(x-w/2,vv.animal_repeatability,w,label="animal repeatability",color="teal")
ax.bar(x+w/2,vv.HTD_var_share_eta2,w,label="herd-test-date share (η²)",color="salmon")
ax.set_xticks(x); ax.set_xticklabels(vv.index); ax.set_ylabel("variance share"); ax.set_ylim(0,1)
ax.set_title("2.4  Variance decomposition (Costa 2020 spirit)"); ax.legend(); ax.grid(alpha=.3,axis="y")
plt.tight_layout(); plt.savefig(FIG/"s2_variance_decomp.png",dpi=130); plt.show()
print("  wrote s2_traits_by_parity / s2_traits_by_SOL / s2_wood_features_hist / s2_variance_decomp")
