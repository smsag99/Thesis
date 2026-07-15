# Clean Analysis Pipeline — Specification (ANALYSIS ONLY)

**Thesis:** Predictive Analytics of Climate Stress Impact on the Italian Mediterranean Buffalo — Physiological and Productive Responses.

**Purpose of this doc:** the full analysis plan for the clean rebuild. Each stage states the exact model/statistics, the variables it uses, the correctness rules, and the literature paper it is validated against. Nothing here is implemented yet; this is the spec to approve before writing code.

**Scope of this pass — ANALYSIS ONLY, no prediction.** The goal is to understand the data rigorously: distributions, relationships, thresholds, exposure windows, and structure (clustering). We do **not** build predictive models in this pass. Every stage is designed to *inform the eventual model choice* — after the analysis we will know the right outcome, the features that matter, the functional form (linear / quadratic / threshold), the best exposure window, and whether discrete groups exist. Predictive modelling (aim 6) and forecasting are explicitly deferred (see "Deferred" at the end).

Data source: `Thesis_Data/Final_Data/After_Processing.csv` (production, cleaned) and `Final_Merged_Data.csv` (+ per-farm daily climate). ~1.65M test-day records · 91,221 buffalo · 329 farms · 2012–2025. Weather is Open-Meteo/ERA5 (proposal states NASA POWER — see Decision D1).

---

## 0. Global correctness rules (invariants for every stage)

| # | Rule | Why |
|---|---|---|
| R1 | **Causal climate effects use the THI *anomaly*** = `THI_1 − mean(THI_1 | farm, calendar-month)`, never raw same-day THI. | Milk and THI both peak in summer for non-heat reasons; raw THI gives the *wrong sign* (heat looks beneficial). |
| R2 | **Analyse summer and winter separately** (or season as a fixed effect). | Annual correlation ≈ 0 because summer (−) and winter (+) cancel. |
| R3 | **Heat exposure is cumulative** — trailing-window means / heat-day counts, not one day. | Milk on a test-day reflects the prior weeks. |
| R4 | **Always control lactation stage (DIM) and parity.** | Longer heat windows correlate with later DIM; uncontrolled, late-lactation SCS masquerades as heat. |
| R5 | **Within-animal demeaning / animal fixed effects; farm-clustered SE.** | Removes between-animal selection; correct SE for clustered data. |
| R6 | **Clustering: standardize (z-score), and interpret without PCA** — cluster on real features (2 at a time for the readable views), `milk_kg` on Y. | Professors' objection: PCA hides which original features drive the grouping. |
| R7 | **`THI_1` is the primary index** (= Vitali 2009 = Matera 2022 formula, verified identical). `THI_2` robustness; `WHI` = cold/wind-chill. | Positive control must use the same index as the reference paper. |
| R8 | **Fixed seeds, pinned versions, logged data-hash** in every notebook. | Reproducibility for the committee. |
| R9 | **Report multiple-testing context** when sweeping many windows/pairs/classes/k. | We run many models; avoid cherry-picking. |

**Key derived variables (exact definitions):**
- `SCS = log2(cells / 100) + 3` (Ali & Shook 1980; verified exact; `cells` in thousands/mL).
- `ECM = ((((fat_p·10) − 40) + ((protein_p·10) − 31))·0.01155 + 1)·milk_kg` — *the FeatureEng formula; verify vs a standard ECM reference (Decision D3).*
- `DIM = dtt − dtc`; `AFC = dtc_first − dtb`.
- `THI_1 = (1.8·T + 32) − (0.55 − 0.0055·RH)·(1.8·T − 26)`, RH in %.
- `THI_anom = THI_1_avg − mean(THI_1_avg | Farm_Code, month)`.
- Trailing window `THI_anom_W = mean(daily THI_anom over prior W days | farm)`, W ∈ {1,3,7,14,30,45,60,90}.
- Stage of lactation (SOL): 12 classes of 30-day DIM.
- Season (calving/sampling): Winter Dec–Feb, Spring Mar–May, Summer Jun–Aug, Autumn Sep–Nov.
- Parity classes: 1, 2, 3, 4, 5+ (cap 6, Costa 2020).
- Region → macro-area: North / Central / South & Islands (ISTAT).

---

## Stage 1 — Data foundation & integrity

**Goal:** one reproducible, audited analysis table; prove production data + weather merge are trustworthy before any inference.

- **1.1 Load & schema** — dtypes, counts, date parsing, duplicate `(Animal_ID, dtt)` check.
- **1.2 Editing** (Costa 2020 + Bobbo): keep `NM==2`; drop `DIM≤0`; `5≤DIM≤400`; parity cap 6; ≥5 test-days per buffalo-within-parity / ≥3 per animal; clip production traits to biological ranges + flag ±3–4 SD for a sensitivity check; log how many rows each rule touches.
- **1.3 Derived variables** — build everything in §0 and persist.
- **1.4 Weather-merge audit** — THI_1_max peaks Jun–Aug (~85), bottoms Jan (~54); unique `(farm,date)` keys; no row inflation; 100% match; SCS identity check.

**Output:** `analysis_base.parquet` + data dictionary.
**Anchor:** Costa 2020 (editing/scale), Bobbo 2023 (editing), our validation notebook.
**Checkpoint:** descriptive means near Costa 2020 (Campania MY ≈ 8.6 kg/d; SCC ≈ 223k cells/mL).

---

## Stage 2 — Descriptive & phenotypic characterization

**Goal:** characterize the population and traits — the "Table 1 / Figure 1" of the thesis.

- **2.1 Descriptive stats by parity × SOL** — mean/SD of MY, FP, PP, SCS, ECM (Matera Table 1; Costa tables).
- **2.2 Lactation curves** — Wood's `y(t)=a·t^b·e^(−ct)` per lactation; derive `peak_DIM=b/c`, `peak_yield=a(b/c)^b e^−b`, `persistency=−(b+1)ln(c)`, `yield_305`; average curve overall + by parity.
- **2.3 Trait correlations** — Pearson **and** Spearman among MY, FP, PP, SCS, ECM, DIM, parity, AFC.
- **2.4 Variance decomposition (Costa 2020):** `milk_kg ~ parity + SOL + month_of_calving + parity:SOL + (1|Animal_ID) + (1|herd_test_date)`; report % variance per level for each trait.

**Anchor:** Costa 2020 (primary), Matera 2022 (Table 1). **Checkpoint:** 3rd parity most productive; PP nadir ~80 DIM; SCS rises through lactation.

---

## Stage 3 — Climate signal & THI–trait relationships

**Goal:** establish, correctly, how THI relates to each trait; reproduce the same-breed positive control.

- **3.1 Season-stratified correlations** — Spearman of each trait vs THI/temp/humidity within summer and within winter (+ full-year to show cancellation); subsets `ET>27°C`, `THI>72` (Piscopo).
- **3.2 THI-class analysis** — Petrocchi bands (<72 optimal / 72–79 mild / 80–89 moderate / ≥90 severe) + Matera's finer classes; LS-means of each trait across classes.
- **3.3 Positive control — Matera 2022 mixed model:**
  - FP/PP/SCS: `y ~ YearSeason_calving + THI_class*parity + SOL*parity + LR + (1|Animal_ID)` (LR = linear covariate on MY when y=SCS, or on SCS when y=FP/PP).
  - MY: same, without LR. LS-means, ANOVA on fixed effects, p<.05.
- **3.4 Causal within-animal anomaly model** (R1/R4/R5): `(trait − animal mean) ~ (THI_anom − animal mean) + DIM_control + parity`, summer subset, farm-clustered SE.
- **3.5 Non-linear shape** — quadratic THI term; cubic-spline/GAM partial dependence of MY & SCS on THI; segmented regression for the inflection point.

**Anchor:** Matera 2022, Piscopo 2024. **Checkpoint:** buffalo THI optimum **59–63**, cold worse than heat, THI×parity significant for PP & SCS; ECM strongly negatively correlated with meteo (Piscopo).

---

## Stage 4 — Heat/cold stress thresholds & exposure  *(proposal aim 5)*

**Goal:** quantify the thresholds and exposure windows at which climate stress affects production/welfare. Still analysis — this produces the aim-5 thresholds, not a model.

- **4.1 THI anomaly** — farm-month normals + anomaly series.
- **4.2 Cumulative/lagged exposure** — regress trait deviation on `THI_anom_W` for each W, DIM+parity controlled, summer, farm-clustered SE; report effect-vs-window profile + **best window per trait**; show raw-vs-DIM-controlled (defend the honest SCS number).
- **4.3 Heat-wave / cold-snap events** — consecutive-day runs: heat wave `THI_1_avg>78` ≥3 d (Petrocchi), severe `>82`; cold snap `THI_1_avg<50` ≥3 d (Matera), severe `temp<0°C`. Within-animal milk in-vs-out; effect by DIM stage; recovery curve; cumulative day-into-event.
- **4.4 Threshold estimation** — segmented/piecewise-linear regression of each trait on THI and on THI_anom → the THI at which MY/SCS/FP/PP inflect (the aim-5 deliverable).
- **4.5 Regional stratification** — causal heat-sensitivity per macro-area (North/Central/South) with cluster-curve + parity controls; forest plots of the +10-THI-anomaly effect (95% CI); Campania+Lazio (DOP core) vs Rest.

**Anchor:** Petrocchi 2023 (bands/definitions), XGBoost heat paper (cumulative windows), Matera 2022 (cold-snap threshold). **Checkpoint:** cold snaps costlier & more frequent than heat waves; severe heat (THI>82) essentially absent; cumulative > same-day.

---

## Stage 5 — Clustering & structure discovery  *(data analysis)*

**Goal:** understand the *structure* of the data — do buffalo/lactations form discrete groups or a continuum? what lactation-curve types exist, and do any map to heat vulnerability? This is exploratory/descriptive analysis that informs later model choice, **not** prediction.

- **5.1 Trait clustering (reproduce Trapanese 2025)** — K-Means vs Hierarchical (Ward's linkage), z-score standardization; choose k by **silhouette + Davies–Bouldin + Calinski–Harabasz**; report cluster profiles. Levels: per-animal (lifetime aggregates) and per-lactation.
- **5.2 Interpretability without PCA** (R6) — cluster on **two real features at a time** (from milk_kg, DIM, parity, AFC, protein_p, fat_p, SCS), plot on the real axes with `milk_kg` on Y; report which pair separates most cleanly and which features discriminate.
- **5.3 Continuum check** — parity-stratified K-Means + **DBSCAN** density check to test whether apparent groups are real or a discreteness artifact (parity/AFC). One dense blob per stratum ⇒ continuum.
- **5.4 Curve-shape clustering** — cluster the interpretable Wood's features (peak, persistency, 305-d) → genuine lactation-curve *types* (e.g. high-peak/low-persistency vs flat/persistent).
- **5.5 Season split** — cluster within summer vs winter to show whether structure itself shifts with season (professors' request).
- **5.6 Link clusters to heat & welfare** — merge THI anomaly + SCS; compare the heat→SCS slope across curve-shape clusters (is there a heat-vulnerable udder-health type?). Connects clustering back to the climate spine.

**Anchor:** Trapanese 2025 (K-means vs hierarchical, silhouette/DBI/CHI). **Checkpoint:** Trapanese silhouette 0.17–0.18, ~2 clusters, DIM+milk the main discriminators — do we replicate the weak-structure/continuum result?

---

## Cross-cutting: literature-comparison protocol

Every stage ends with a comparison cell: (a) name the published number/figure, (b) compute our equivalent, (c) mark match / partial / divergence and the likely reason (breed, farm count, weather source, method). Divergences are findings, not failures.

## Proposed file structure

```
codes_clean/
  01_data_foundation.ipynb
  02_descriptive_characterization.ipynb
  03_thi_trait_relationships.ipynb
  04_thresholds_exposure.ipynb
  05_clustering_structure.ipynb
  lit_comparison.md   # running table of our-vs-paper results
```

---

## Deferred (explicitly NOT in this analysis pass)

Per the analysis-only scope, these wait until the analysis tells us the right target/features/shape/window:

- **Predictive models (aim 6)** — production-loss regression + high-SCC classification (Bobbo 2023a/b, XGBoost templates). Stage 3–5 results (which features matter, linear vs threshold shape, best exposure window, whether groups exist, split-by-animal for leakage) will drive model & feature selection.
- **HTS forecasting** — Total→Farm→Animal milk forecasting with reconciliation (Hyndman, Wickramasuriya). Off-proposal secondary.
- **AFC production/economics** — optional analysis chapter (mixed models by AFC class; Calanni Macchio 2025, Santinello 2025) — can slot in as analysis if wanted (Decision D5).

## Open decisions to confirm before coding

- **D1 — Weather source:** keep Open-Meteo/ERA5 (already merged) or re-extract NASA POWER to match the proposal?
- **D2 — Season definition:** calendar warm-half (Apr–Sep) vs strict summer (Jun–Sep)?
- **D3 — ECM formula:** verify the FeatureEng ECM formula against a standard reference before using ECM.
- **D4 — Clustering scope:** which levels (animal / lactation / test-day) and how far to push 5.4–5.6; reproduce Trapanese exactly + our refinements, or the refinements only?
- **D5 — AFC analysis chapter:** include now (as analysis) or defer?
