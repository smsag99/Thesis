# `codes/` — analysis notebooks

Each analysis area has **one consolidated notebook** at the top of its folder, with the original
working notebooks it was built from archived in `raw/`. Data paths are relative to each notebook's
folder: the consolidated notebooks read `../../Thesis_Data/`, the `raw/` copies read `../../../Thesis_Data/`.

| Folder | Consolidated notebook | Built from (`raw/`) |
|---|---|---|
| `climate_signal/` | **`Climate_Signal_Analysis.ipynb`** | Climate_Signal_Validation, Correlation_analysis, NonLinear_Correlation, Lagged_Heat_Effects, THI_AnimalVar_Averages |
| `clustering/` | **`Clustering_Analysis.ipynb`** | Clustering, Clustering_TimeSeries, Clustering_FeaturePairs, Clustering_ParityStratified, Clustering_CurveShape_Heat, FeaturePairs_Season_Climate |
| `forecasting/` | **`Hierarchical_Forecasting.ipynb`** | HTS, HTS_Forecasting, Hierarchical, Hierchical_Structure |
| `data/` | *(not consolidated — kept as-is)* | dataExploring, FeatureEng_v2, WeatherDataDownloader, COEF |

## `climate_signal/Climate_Signal_Analysis.ipynb`
The climate → production/welfare story, ordered as the actual investigation:
1. Data confidence & weather-merge audit → 2. Linear correlations & mutual information →
3. The **season confound** (annual r ≈ 0) → 4. The fix: **THI anomaly** + Matera-2022 positive control
(milk ↓, SCS ↑) → 5. Within-animal effects & LMM variance structure → 6. Cumulative/lagged heat
exposure (best window per trait) → 7. Heat-wave/cold-snap events → 8. Regional heat sensitivity
(cluster-curve controlled) → 9. Farm heterogeneity.

## `clustering/Clustering_Analysis.ipynb`
Does the population form discrete groups? 1. Baseline point-trait clustering (K-Means / Hierarchical /
DBSCAN) → 2. No-PCA feature pairs → 3. Removing the parity confound (→ **continuum**) →
4. Season & climate feature pairs → 5. **Curve-shape clustering** (the real structure, via Wood's
features) → 6. Linking curve types to heat & udder health (SCS).

## `forecasting/Hierarchical_Forecasting.ipynb`
Secondary / off-proposal track. **Part A** — reconciled `Total → Farm → Animal` milk forecasting
(AutoETS/SeasonalNaive base forecasts → BottomUp/MinTrace reconciliation → evaluation → CV).
**Part B** — Wood's lactation-curve tensors + fit-quality report. Needs `hierarchicalforecast` and
`statsforecast` (the notebook's first cell `pip install`s them).

## `data/` — building & ingesting the dataset
`dataExploring` (raw look, Colab), `FeatureEng_v2` (cleaning + derived vars → `After_Processing.csv`),
`WeatherDataDownloader` (Open-Meteo → THI indices, Colab), `COEF` (data/AFC checks). Left as-is.

---
*Environment note:* the analysis stack (scipy ≥ 1.16 / scikit-learn / statsmodels) runs under the
miniforge Python. The forecasting libraries want an older scipy — give them a separate env.
