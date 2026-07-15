# %% [markdown]
# # Stage 6 — Regional Clustering  (data analysis, no prediction)
#
# Motivation: Stage 4's regional forest showed the heat→SCS signal is regional (strongest in the
# North) and Stage 5 found production is a continuum, not discrete animal types. The dataset now
# carries `macro_area` / `Region` / `Province`. Question here: **is the geography a real structure?**
# Do farms/regions form groups by their production + climate profile, and does that grouping recover
# the North/Central/South split — or is region just a climate-exposure gradient?
#
# 1. Regional overview — coverage, imbalance (Campania+Lazio dominate).
# 2. Region-level profiles — interpretable mean trait+climate table per region.
# 3. Region-level clustering — Ward dendrogram + K-Means on region profiles; compare to macro_area.
# 4. Farm-level clustering (n=316) — no PCA; does the farm profile recover macro_area (ARI)?
#    Climate-only vs production-only: which drives the geography?
# 5. Regional heat-response clustering — per-region within-animal summer THI slopes (milk & SCS),
#    reproduce the Stage-4 regional forest, then group regions into heat-vulnerable types.
# 6. Season split — does farm structure shift summer vs winter.
#
# Anchors: Stage 4 regional_forest (North SCS +0.41/+10, 30d), Trapanese continuum (Stage 5).
# Checkpoint: climate (THI) separates macro_area strongly; North = coolest + highest SCS-heat slope.

# %%
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
SEED = 42

def find_root():
    cands = [Path("Thesis_Data"), Path("../../Thesis_Data")]
    try: cands.append(Path(__file__).resolve().parents[2] / "Thesis_Data")
    except NameError: pass
    for c in cands:
        if c.exists(): return c.resolve().parent
    raise SystemExit("no Thesis_Data")
ROOT = find_root(); OUT = ROOT / "codes" / "pipeline" / "outputs"
FIG = OUT / "figures"; TAB = OUT / "tables"; FIG.mkdir(exist_ok=True, parents=True); TAB.mkdir(exist_ok=True, parents=True)
def rule(m): print("\n" + "=" * 78 + f"\n{m}\n" + "=" * 78)
def verdict(n, ok, d=""): print(f"  [{'MATCH' if ok else 'CHECK'}] {n}" + (f"  --  {d}" if d else ""))
def sil(X, lab): return round(silhouette_score(X, lab), 3) if len(set(lab)) > 1 else np.nan
def best_k(X, ks=range(2, 7)):
    b = (None, -1, None)
    for k in ks:
        if k >= len(X): break
        lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
        s = sil(X, lab)
        if s > b[1]: b = (k, s, lab)
    return b

base = pd.read_parquet(OUT / "analysis_base.parquet")
rule("LOADED")
print(f"  records={len(base):,} | animals={base.Animal_ID.nunique():,} | farms={base.Farm_Code.nunique()} "
      f"| regions={base.Region.nunique()} | macro_areas={base.macro_area.nunique()}")

# %% [markdown]
# ## 6.1 Regional overview — coverage & imbalance
# The panel is heavily South-weighted (Campania) with a much smaller Northern tail. Any regional
# clustering has to be read with this imbalance in mind: small regions = noisy profiles.

# %%
rule("6.1  REGIONAL OVERVIEW")
cov = (base.groupby(["macro_area", "Region"])
       .agg(farms=("Farm_Code", "nunique"), animals=("Animal_ID", "nunique"),
            records=("milk_kg", "size"))
       .reset_index().sort_values("records", ascending=False))
cov["rec_share_%"] = (100 * cov.records / cov.records.sum()).round(1)
cov.to_csv(TAB / "region_coverage.csv", index=False)
print(cov.to_string(index=False))
print("\n  macro_area totals:")
print(base.groupby("macro_area").agg(farms=("Farm_Code", "nunique"),
      animals=("Animal_ID", "nunique"), records=("milk_kg", "size")).to_string())

# %% [markdown]
# ## 6.2 Region-level trait + climate profiles (interpretable, no PCA)
# One row per region: mean production, quality, herd structure and climate exposure. This is the
# feature table the region clustering runs on.

# %%
rule("6.2  REGION-LEVEL PROFILES")
def heatwave_frac(s): return float((s >= 78).mean())  # THI_1_max >= 78 = heat-stress day (buffalo)
reg = (base.groupby("Region")
       .agg(farms=("Farm_Code", "nunique"), records=("milk_kg", "size"),
            milk_kg=("milk_kg", "mean"), fat_p=("fat_p", "mean"), protein_p=("protein_p", "mean"),
            SCS=("SCS", "mean"), ECM=("ECM", "mean"),
            THI_avg=("THI_1_avg", "mean"), THI_max=("THI_1_max", "mean"),
            parity=("parity", "mean"), AFC_mo=("true_AFC_days", lambda s: s.mean() / 30.44))
       .join(base.groupby("Region").THI_1_max.apply(heatwave_frac).rename("heatday_frac")))
reg["macro_area"] = base.groupby("Region").macro_area.first()
reg = reg.round(3)
reg.to_csv(TAB / "region_profiles.csv")
CLUS_FEATS = ["milk_kg", "fat_p", "protein_p", "SCS", "THI_avg", "THI_max", "heatday_frac", "parity", "AFC_mo"]
print(reg[["macro_area", "farms", "records"] + CLUS_FEATS].to_string())

# %% [markdown]
# ## 6.3 Region-level clustering — Ward dendrogram + K-Means (does it recover macro_area?)
# Only well-sampled regions (>=2 farms AND >=5000 records) — a 1-farm region is one herd, not a
# region. Small N (points = regions) so the dendrogram is the honest tool; K-Means is a cross-check.

# %%
rule("6.3  REGION-LEVEL CLUSTERING")
regC = reg[(reg.farms >= 2) & (reg.records >= 5000)].copy()
print(f"  clustering {len(regC)} well-sampled regions: {list(regC.index)}")
Xr = StandardScaler().fit_transform(regC[CLUS_FEATS])
Z = linkage(Xr, method="ward")
for k in (2, 3, 4):
    lab = fcluster(Z, k, criterion="maxclust")
    ari = adjusted_rand_score(regC.macro_area, lab)
    print(f"  Ward cut k={k}: silhouette={sil(Xr, lab)}  ARI vs macro_area={round(ari, 3)}  "
          f"sizes={np.bincount(lab)[1:].tolist()}")
k_reg, sil_reg, lab_reg = best_k(Xr, ks=range(2, 6))
regC["region_cl"] = fcluster(Z, 3, criterion="maxclust")
print(f"\n  K-Means best k={k_reg} (silhouette={sil_reg})")
print("\n  Ward k=3 clusters vs macro_area:")
print(pd.crosstab(regC.region_cl, regC.macro_area).to_string())
prof3 = regC.groupby("region_cl")[["milk_kg", "SCS", "THI_avg", "THI_max", "heatday_frac"]].mean().round(2)
prof3["regions"] = regC.groupby("region_cl").apply(lambda d: ", ".join(d.index))
print("\n  cluster profiles:"); print(prof3.to_string())

# %% [markdown]
# ## 6.4 Farm-level clustering (n=316, no PCA) — is geography recoverable?
# One row per farm. Cluster on the full profile, then on climate-only and production-only feature
# blocks, and measure how well each recovers macro_area (Adjusted Rand Index). Tells us whether the
# North/South split is a *climate* fact or a *production* fact.

# %%
rule("6.4  FARM-LEVEL CLUSTERING")
farm = (base.groupby("Farm_Code")
        .agg(records=("milk_kg", "size"),
             milk_kg=("milk_kg", "mean"), fat_p=("fat_p", "mean"), protein_p=("protein_p", "mean"),
             SCS=("SCS", "mean"), THI_avg=("THI_1_avg", "mean"), THI_max=("THI_1_max", "mean"),
             parity=("parity", "mean"), AFC_mo=("true_AFC_days", lambda s: s.mean() / 30.44))
        .join(base.groupby("Farm_Code").THI_1_max.apply(heatwave_frac).rename("heatday_frac")))
farm["macro_area"] = base.groupby("Farm_Code").macro_area.first()
farm["Region"] = base.groupby("Farm_Code").Region.first()
farm = farm[farm.records >= 200].dropna()   # drop tiny farms with unstable means
print(f"  farms clustered: {len(farm)}")
blocks = {
    "full":       ["milk_kg", "fat_p", "protein_p", "SCS", "THI_avg", "THI_max", "heatday_frac", "parity", "AFC_mo"],
    "climate":    ["THI_avg", "THI_max", "heatday_frac"],
    "production": ["milk_kg", "fat_p", "protein_p", "SCS"],
}
frows = []
for name, feats in blocks.items():
    X = StandardScaler().fit_transform(farm[feats])
    k, s, lab = best_k(X, ks=range(2, 7))
    ari3 = adjusted_rand_score(farm.macro_area, KMeans(3, n_init=10, random_state=SEED).fit_predict(X))
    frows.append({"block": name, "best_k": k, "silhouette": round(s, 3),
                  "ARI_k=bestk": round(adjusted_rand_score(farm.macro_area, lab), 3),
                  "ARI_k=3_vs_macroarea": round(ari3, 3), "DBI": round(davies_bouldin_score(X, lab), 3)})
    if name == "full":
        farm["farm_cl"] = lab; k_full = k
ff = pd.DataFrame(frows); ff.to_csv(TAB / "farm_cluster_blocks.csv", index=False)
print(ff.to_string(index=False))
print("\n  full-profile clusters vs macro_area:")
print(pd.crosstab(farm.farm_cl, farm.macro_area).to_string())
fp = farm.groupby("farm_cl")[blocks["full"]].mean().round(2)
fp["n_farms"] = farm.farm_cl.value_counts()
print("\n  farm-cluster profiles:"); print(fp.to_string())

# %% [markdown]
# ## 6.5 Regional heat-response clustering — reproduce Stage-4 forest, then group regions
# Per region: within-animal, DIM-controlled summer slope of milk & SCS on THI_anom (per +10). Same
# spec as Stage 4/5. Then cluster regions on (milk-slope, SCS-slope) to name heat-response types.

# %%
rule("6.5  REGIONAL HEAT-RESPONSE")
Sm = base[base.is_summer_warmhalf].copy(); Sm["DIM2"] = Sm.DIM.astype(float) ** 2
def region_slope(sub, trait, min_animals=200):
    d = sub[[trait, "THI_anom", "DIM", "DIM2", "Animal_ID", "Farm_Code"]].dropna().copy()
    if d.Animal_ID.nunique() < min_animals: return None
    g = d.groupby("Animal_ID")
    for c in [trait, "THI_anom", "DIM", "DIM2"]:
        d[c + "_w"] = d[c] - g[c].transform("mean")
    m = smf.ols(f"{trait}_w ~ THI_anom_w + DIM_w + DIM2_w - 1", data=d)
    # farm-clustered SE needs >=2 farms; single-farm regions fall back to heteroskedasticity-robust
    if d.Farm_Code.nunique() >= 2:
        r = m.fit(cov_type="cluster", cov_kwds={"groups": d.Farm_Code})
    else:
        r = m.fit(cov_type="HC1")
    return round(10 * r.params["THI_anom_w"], 4), round(10 * 1.96 * r.bse["THI_anom_w"], 4), int(d.Animal_ID.nunique())
hrows = []
for rg in Sm.Region.unique():
    sub = Sm[Sm.Region == rg]
    row = {"Region": rg, "macro_area": sub.macro_area.iloc[0], "n_animals_summer": sub.Animal_ID.nunique()}
    ok = True
    for t in ["milk_kg", "SCS"]:
        res = region_slope(sub, t)
        if res is None: ok = False; break
        row[f"{t}_per10"], row[f"{t}_ci95"] = res[0], res[1]
    if ok: hrows.append(row)
hh = pd.DataFrame(hrows).sort_values("SCS_per10", ascending=False)
hh.to_csv(TAB / "regional_heat_slopes.csv", index=False)
print(hh.to_string(index=False))
# cluster regions on (milk slope, SCS slope)
Xh = StandardScaler().fit_transform(hh[["milk_kg_per10", "SCS_per10"]])
k_h, s_h, lab_h = best_k(Xh, ks=range(2, 5))
hh["heat_cl"] = lab_h
print(f"\n  heat-response clusters k={k_h} (silhouette={s_h}):")
print(hh.groupby("heat_cl")[["milk_kg_per10", "SCS_per10"]].mean().round(3).to_string())
print("  regions per heat cluster:")
for c in sorted(hh.heat_cl.unique()):
    print(f"    cluster {c}: {', '.join(hh[hh.heat_cl == c].Region)}")

# %% [markdown]
# ## 6.6 Season split — does farm structure shift summer vs winter?

# %%
rule("6.6  SEASON SPLIT (farm clustering)")
for name, mask in [("summer", base.is_summer_warmhalf), ("winter", ~base.is_summer_warmhalf)]:
    fm = base[mask].groupby("Farm_Code").agg(
        milk_kg=("milk_kg", "mean"), SCS=("SCS", "mean"),
        THI_avg=("THI_1_avg", "mean"), THI_max=("THI_1_max", "mean")).dropna()
    ma = base[mask].groupby("Farm_Code").macro_area.first().reindex(fm.index)
    X = StandardScaler().fit_transform(fm)
    k, s, lab = best_k(X, ks=range(2, 6))
    ari = adjusted_rand_score(ma, KMeans(3, n_init=10, random_state=SEED).fit_predict(X))
    print(f"  {name}: farms={len(fm)}  best k={k}  silhouette={s}  ARI(k=3) vs macro_area={round(ari, 3)}")

# %% [markdown]
# ## 6.7 Literature / cross-stage checkpoint

# %%
rule("6.7  CHECKPOINT")
ari_full = ff.loc[ff.block == "full", "ARI_k=3_vs_macroarea"].iloc[0]
ari_clim = ff.loc[ff.block == "climate", "ARI_k=3_vs_macroarea"].iloc[0]
ari_prod = ff.loc[ff.block == "production", "ARI_k=3_vs_macroarea"].iloc[0]
sil_clim = ff.loc[ff.block == "climate", "silhouette"].iloc[0]
sil_prod = ff.loc[ff.block == "production", "silhouette"].iloc[0]
verdict("Climate exposure forms cleaner farm groups than production traits", sil_clim > sil_prod,
        f"climate silhouette={sil_clim} vs production {sil_prod}")
verdict("Geography tracks CLIMATE more than production (a climate gradient, not a production type)",
        ari_clim >= ari_prod, f"climate ARI={ari_clim} vs production ARI={ari_prod}")
north_scs = hh.loc[hh.macro_area == "North", "SCS_per10"]
verdict("Stage-4 reproduced: North has among the strongest SCS heat slope",
        len(north_scs) > 0 and north_scs.max() >= hh.SCS_per10.median(),
        f"North SCS_per10={north_scs.round(3).tolist()} (median {round(hh.SCS_per10.median(), 3)})")
summary = {
    "regions_total": int(base.Region.nunique()), "regions_wellsampled": int(len(regC)),
    "farms_clustered": int(len(farm)), "farm_best_k": int(k_full),
    "ARI_full": float(ari_full), "ARI_climate": float(ari_clim), "ARI_production": float(ari_prod),
    "silhouette_climate": float(sil_clim), "silhouette_production": float(sil_prod),
    "region_heat_clusters": int(k_h),
    "region_SCS_slopes": hh.set_index("Region").SCS_per10.to_dict(),
}
(TAB / "stage6_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("\n  summary:", json.dumps(summary, indent=2, default=str))

# %% [markdown]
# ## 6.8 Figures — regional structure

# %%
rule("6.8  FIGURES")
# Fig 6.1: region dendrogram, leaves colored/labelled by macro_area
fig, ax = plt.subplots(figsize=(11, 5))
lbl = [f"{r} [{regC.loc[r, 'macro_area'][0]}]" for r in regC.index]
dendrogram(Z, labels=lbl, ax=ax, leaf_rotation=90, color_threshold=0.7 * Z[:, 2].max())
ax.set_title("6.1  Region dendrogram (Ward) — leaf tag = macro_area [N/C/S]")
ax.set_ylabel("Ward distance"); plt.tight_layout(); plt.savefig(FIG / "s6_region_dendrogram.png", dpi=130); plt.show()

# Fig 6.2: farms on climate (THI_avg) × production (milk), colored by macro_area
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
col = {"North": "steelblue", "Central": "seagreen", "South": "firebrick"}
for m, c in col.items():
    d = farm[farm.macro_area == m]
    ax[0].scatter(d.THI_avg, d.milk_kg, c=c, s=22, alpha=.7, label=m)
ax[0].set_xlabel("mean THI_1_avg (climate exposure)"); ax[0].set_ylabel("mean milk_kg")
ax[0].set_title("6.2  Farms: climate × production (color = macro_area)"); ax[0].legend(); ax[0].grid(alpha=.3)
sc = ax[1].scatter(farm.THI_avg, farm.milk_kg, c=farm.farm_cl, cmap="viridis", s=22, alpha=.7)
ax[1].set_xlabel("mean THI_1_avg"); ax[1].set_ylabel("mean milk_kg")
ax[1].set_title(f"6.2  Same farms colored by K-Means cluster (k={k_full})"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "s6_farm_clusters.png", dpi=130); plt.show()

# Fig 6.3: regional heat forest — SCS & milk slope per +10 THI anomaly, ordered by SCS slope
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, (t, cc) in zip(ax, [("SCS", "firebrick"), ("milk_kg", "steelblue")]):
    s = hh.sort_values(f"{t}_per10"); yv = range(len(s))
    a.errorbar(s[f"{t}_per10"], yv, xerr=s[f"{t}_ci95"], fmt="o", capsize=3, color=cc)
    a.axvline(0, c="grey", lw=.8); a.set_yticks(list(yv))
    a.set_yticklabels([f"{r} [{m[0]}]" for r, m in zip(s.Region, s.macro_area)])
    a.set_title(f"6.3  {t}: within-animal summer heat slope per +10 THI-anom"); a.set_xlabel("Δ per +10"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "s6_regional_heat_forest.png", dpi=130); plt.show()

# Fig 6.4: heat-response map — milk slope vs SCS slope, colored by heat cluster, sized by n_animals
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(hh.milk_kg_per10, hh.SCS_per10, c=hh.heat_cl, cmap="coolwarm",
                s=30 + hh.n_animals_summer / hh.n_animals_summer.max() * 300, alpha=.75, edgecolor="k")
for _, r in hh.iterrows():
    ax.annotate(r.Region, (r.milk_kg_per10, r.SCS_per10), fontsize=8, xytext=(4, 2), textcoords="offset points")
ax.axhline(0, c="grey", lw=.8); ax.axvline(0, c="grey", lw=.8)
ax.set_xlabel("milk heat slope (Δ per +10)"); ax.set_ylabel("SCS heat slope (Δ per +10)")
ax.set_title("6.4  Regional heat-response map (top-left = heat-vulnerable: milk↓ & SCS↑)")
ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG / "s6_heat_response_map.png", dpi=130); plt.show()
print("  wrote s6_region_dendrogram / s6_farm_clusters / s6_regional_heat_forest / s6_heat_response_map")
rule("STAGE 6 COMPLETE")
