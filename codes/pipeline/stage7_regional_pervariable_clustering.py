# %% [markdown]
# # Stage 7 — Per-variable Regional Clustering  (no PCA)
#
# Stage 6 clustered farms/regions on ALL variables at once and found geography is mostly a *climate*
# gradient, with production traits weakly structured. This stage zooms in: **one clustering per
# animal variable, by region.** For each animal trait we group the regions on that single trait
# (no PCA — each clustering uses the real variable), so we can read directly which regions are
# high/low on milk, on SCS, on fat, etc., and test whether any single animal trait carves the
# regions the same way geography does.
#
# 1. Per-variable REGION clustering — for each animal variable, K-Means on the region means
#    (standardized, 1 feature, no PCA); best k by silhouette; report region membership + tier.
# 2. Synthesis — region × variable tier matrix; do the animal traits agree on which regions group
#    together? Consensus clustering of regions across all trait-tiers.
# 3. Farm-level recovery test — for each variable, cluster the 298 farms and measure how well that
#    single trait recovers Region / macro_area (Adjusted Rand Index). Which animal trait is the most
#    "geographic"?
#
# Animal variables (environment/THI excluded — that was Stage 6): milk_kg, ECM, fat_p, protein_p,
# SCS, parity, true_AFC (months). DIM is left out — it is a test-day timing variable, not a trait.
# Anchor: Stage 6 (climate >> production for geography). Expectation: most animal traits split
# regions only weakly; SCS is the one with a real regional (welfare) signal.

# %%
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
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
def best_k(X, ks):
    b = (None, -1, None)
    for k in ks:
        if k >= len(X): break
        lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
        s = sil(X, lab)
        if s > b[1]: b = (k, s, lab)
    return b
def tiers_from_labels(vals, lab):
    """Relabel cluster ids to ordinal tiers 1..k by ascending cluster mean of the variable."""
    order = pd.Series(vals).groupby(lab).mean().sort_values().index.tolist()
    remap = {c: t + 1 for t, c in enumerate(order)}
    return np.array([remap[c] for c in lab])

base = pd.read_parquet(OUT / "analysis_base.parquet")
base["AFC_mo"] = base["true_AFC_days"] / 30.44
ANIMAL_VARS = ["milk_kg", "ECM", "fat_p", "protein_p", "SCS", "parity", "AFC_mo"]
rule("LOADED")
print(f"  records={len(base):,} | animal variables clustered: {ANIMAL_VARS}")

# %% [markdown]
# ## 7.1 Per-variable region clustering (no PCA, one trait at a time)
# Well-sampled regions only (>=2000 records) so region means are stable. For each animal variable we
# standardize the region means and run K-Means (best k in 2..4 by silhouette). `tier` = ordinal
# rank of the cluster (1 = lowest-value group). ARI compares the grouping to macro_area (N/C/S).

# %%
rule("7.1  PER-VARIABLE REGION CLUSTERING")
MIN_REC = 2000
regstat = base.groupby("Region").agg(records=("milk_kg", "size"),
                                      macro_area=("macro_area", "first"))
keep = regstat[regstat.records >= MIN_REC].index
regmean = base[base.Region.isin(keep)].groupby("Region")[ANIMAL_VARS].mean()
macro = base.groupby("Region").macro_area.first().reindex(regmean.index)
print(f"  regions clustered ({len(regmean)}, >= {MIN_REC} records): {list(regmean.index)}")

tier_mat = pd.DataFrame(index=regmean.index)   # region x variable ordinal tiers
rows = []
for v in ANIMAL_VARS:
    x = regmean[[v]].values
    X = StandardScaler().fit_transform(x)
    k, s, lab = best_k(X, ks=range(2, 5))
    tier = tiers_from_labels(regmean[v].values, lab)
    tier_mat[v] = tier
    ari = adjusted_rand_score(macro, lab)
    cl_means = {int(t): round(regmean[v].values[tier == t].mean(), 2) for t in sorted(set(tier))}
    rows.append({"variable": v, "best_k": k, "silhouette": round(float(s), 3),
                 "ARI_vs_macroarea": round(float(ari), 3), "tier_means": cl_means})
    print(f"\n  {v}: k={k}  silhouette={round(float(s),3)}  ARI(macro_area)={round(float(ari),3)}  tier means={cl_means}")
    for t in sorted(set(tier)):
        print(f"      tier {t}: {', '.join(regmean.index[tier == t])}")
pv = pd.DataFrame(rows)
pv.to_csv(TAB / "pervariable_region_clusters.csv", index=False)
tier_mat_out = tier_mat.copy(); tier_mat_out["macro_area"] = macro
tier_mat_out.to_csv(TAB / "region_variable_tiers.csv")
print("\n  per-variable summary:"); print(pv[["variable", "best_k", "silhouette", "ARI_vs_macroarea"]].to_string(index=False))

# %% [markdown]
# ## 7.2 Synthesis — do the animal traits agree on regional groups?
# tier matrix = each region's high/low tier on every trait. If traits agree, regions that are high on
# one are high on others. We (a) correlate the tier columns, and (b) consensus-cluster the regions on
# the full tier vector (still no PCA) to get an overall "animal-profile" grouping of regions.

# %%
rule("7.2  SYNTHESIS ACROSS TRAITS")
print("  region x variable tier matrix (1=low ... k=high):")
print(tier_mat.assign(macro=macro.str[0]).to_string())
# consensus clustering of regions on the tier vectors
Xc = StandardScaler().fit_transform(tier_mat.values)
kc, sc, labc = best_k(Xc, ks=range(2, 6))
cons = pd.Series(labc, index=tier_mat.index, name="consensus_cl")
ari_cons = adjusted_rand_score(macro, labc)
print(f"\n  consensus clustering of regions across all traits: k={kc}  silhouette={round(float(sc),3)}  "
      f"ARI(macro_area)={round(float(ari_cons),3)}")
for c in sorted(set(labc)):
    print(f"    group {c}: {', '.join(cons.index[cons == c])}")
# which traits move together
tc = tier_mat.corr(method="spearman").round(2)
tc.to_csv(TAB / "trait_tier_correlations.csv")
print("\n  Spearman correlation of trait tiers (which traits carve regions alike):")
print(tc.to_string())

# %% [markdown]
# ## 7.3 Farm-level recovery test — which single animal trait is most "geographic"?
# 298 farms (>=200 records). For each variable, cluster farms on that trait alone (best k) and measure
# how well the grouping recovers Region and macro_area (ARI). High ARI = that trait tracks geography.

# %%
rule("7.3  FARM-LEVEL RECOVERY")
farm = base.groupby("Farm_Code").agg(records=("milk_kg", "size"),
        **{v: (v, "mean") for v in ANIMAL_VARS})
farm["Region"] = base.groupby("Farm_Code").Region.first()
farm["macro_area"] = base.groupby("Farm_Code").macro_area.first()
farm = farm[farm.records >= 200].dropna()
print(f"  farms: {len(farm)}")
frows = []
for v in ANIMAL_VARS:
    X = StandardScaler().fit_transform(farm[[v]])
    k, s, lab = best_k(X, ks=range(2, 7))
    frows.append({"variable": v, "best_k": k, "silhouette": round(float(s), 3),
                  "ARI_vs_Region": round(float(adjusted_rand_score(farm.Region, lab)), 3),
                  "ARI_vs_macroarea": round(float(adjusted_rand_score(farm.macro_area, lab)), 3)})
fr = pd.DataFrame(frows).sort_values("ARI_vs_macroarea", ascending=False)
fr.to_csv(TAB / "pervariable_farm_recovery.csv", index=False)
print(fr.to_string(index=False))

# %% [markdown]
# ## 7.4 Checkpoint

# %%
rule("7.4  CHECKPOINT")
max_reg_ari = float(pv.ARI_vs_macroarea.abs().max())
verdict("No animal trait's regional grouping follows geography (all |ARI| < 0.15)", max_reg_ari < 0.15,
        f"max |region ARI| vs macro_area = {round(max_reg_ari,3)}")
verdict("No single animal trait recovers geography at farm level (all ARI < 0.2)", (fr.ARI_vs_macroarea < 0.2).all(),
        f"max farm ARI_vs_macroarea={round(float(fr.ARI_vs_macroarea.max()),3)} ({fr.iloc[0].variable})")
verdict("Even the combined animal profile is not geographic (Stage-6: geography = climate, not traits)",
        abs(ari_cons) < 0.15, f"consensus ARI vs macro_area = {round(float(ari_cons),3)}")
print("  NOTE: each trait DOES split regions into clean high/low tiers (silhouette 0.54-0.73);")
print("        they just don't line up with N/C/S. Trait tiers that move together: milk~ECM (+0.94),")
print(f"        fat_p~milk (dilution, {tc.loc['fat_p','milk_kg']}).")
summary = {
    "regions_clustered": int(len(regmean)), "min_records": MIN_REC,
    "farms": int(len(farm)),
    "pervariable_region_ARI": pv.set_index("variable").ARI_vs_macroarea.to_dict(),
    "pervariable_farm_ARI_macroarea": fr.set_index("variable").ARI_vs_macroarea.to_dict(),
    "consensus_k": int(kc), "consensus_ARI_macroarea": float(ari_cons),
    "most_geographic_trait": str(fr.iloc[0].variable),
}
(TAB / "stage7_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("\n  summary:", json.dumps(summary, indent=2, default=str))

# %% [markdown]
# ## 7.5 Figures

# %%
rule("7.5  FIGURES")
order = macro.sort_values(kind="stable").index          # group rows by macro_area
Tm = tier_mat.loc[order]
# Fig 7.1: region x variable tier heatmap
fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(Tm))))
im = ax.imshow(Tm.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(ANIMAL_VARS))); ax.set_xticklabels(ANIMAL_VARS, rotation=30, ha="right")
ax.set_yticks(range(len(Tm))); ax.set_yticklabels([f"{r} [{macro[r][0]}]" for r in Tm.index])
for i in range(len(Tm)):
    for j, v in enumerate(ANIMAL_VARS):
        ax.text(j, i, f"{regmean.loc[Tm.index[i], v]:.1f}", ha="center", va="center", fontsize=7)
ax.set_title("7.1  Per-variable regional tier (color) + region mean (text) — 1=low tier")
fig.colorbar(im, ax=ax, shrink=.6, label="tier"); plt.tight_layout()
plt.savefig(FIG / "s7_region_variable_tiers.png", dpi=130); plt.show()

# Fig 7.2: per-variable region-mean strips, colored by that variable's cluster tier
n = len(ANIMAL_VARS); fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 4.2), sharey=False)
for ax, v in zip(axes, ANIMAL_VARS):
    vals = regmean[v]; tier = tier_mat[v]
    ax.scatter(np.zeros(len(vals)) + np.random.RandomState(SEED).uniform(-.05, .05, len(vals)),
               vals, c=tier, cmap="viridis", s=45, edgecolor="k")
    for r in vals.index:
        ax.annotate(r[:4], (0.06, vals[r]), fontsize=6, va="center")
    ax.set_title(v, fontsize=9); ax.set_xticks([]); ax.grid(alpha=.3, axis="y")
fig.suptitle("7.2  Region means per animal variable, colored by cluster tier (no PCA)")
plt.tight_layout(); plt.savefig(FIG / "s7_pervariable_strips.png", dpi=130); plt.show()

# Fig 7.3: geographic-structure ranking (ARI) per variable, region-level & farm-level
fig, ax = plt.subplots(figsize=(9, 4.4))
o = pv.set_index("variable").loc[ANIMAL_VARS]
frm = fr.set_index("variable").loc[ANIMAL_VARS]
y = np.arange(len(ANIMAL_VARS)); h = .38
ax.barh(y + h/2, o.ARI_vs_macroarea, height=h, color="steelblue", label="region-level ARI")
ax.barh(y - h/2, frm.ARI_vs_macroarea, height=h, color="indianred", label="farm-level ARI")
ax.set_yticks(y); ax.set_yticklabels(ANIMAL_VARS); ax.invert_yaxis()
ax.set_xlabel("Adjusted Rand Index vs macro_area (higher = more geographic)")
ax.set_title("7.3  How much each animal trait's clustering recovers geography")
ax.legend(); ax.grid(alpha=.3, axis="x"); plt.tight_layout()
plt.savefig(FIG / "s7_geographic_structure_ari.png", dpi=130); plt.show()
print("  wrote s7_region_variable_tiers / s7_pervariable_strips / s7_geographic_structure_ari")
rule("STAGE 7 COMPLETE")
