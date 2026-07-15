# `codes/` — analysis notebooks

Notebooks are grouped by purpose. All data paths are relative to each notebook's
folder, i.e. they read from `../../Thesis_Data/` (two levels up).

> The Colab-era notebooks (`data/dataExploring`, `data/WeatherDataDownloader`)
> instead point at `/content/drive/MyDrive/Thesis_Data/` and are meant to run on
> Google Colab.

## `data/` — building & ingesting the dataset
| Notebook | What it does |
|---|---|
| `dataExploring.ipynb` | First look at the raw `Dati_ok.txt` export (Colab). |
| `FeatureEng_v2.ipynb` | Cleaning + derived variables (SCS, ECM, DIM, AFC …) → `After_Processing.csv`. |
| `WeatherDataDownloader.ipynb` | Pulls Open-Meteo weather per farm → `THI_1/THI_2/WHI` indices (Colab). |
| `COEF.ipynb` | Small data / AFC-fix and coefficient checks. |

## `climate_signal/` — climate (THI) → trait relationships
| Notebook | What it does |
|---|---|
| `Correlation_analysis.ipynb` | Main climate analysis: Pearson/Spearman, mutual information, splines, animal fixed-effects, LMM, heat-wave/cold-snap events. |
| `NonLinear_Correlation.ipynb` | Quadratic THI (Costa 2020), season-stratified correlations, breakpoints. |
| `Lagged_Heat_Effects.ipynb` | Cumulative / lagged multi-day heat-exposure windows. |
| `THI_AnimalVar_Averages.ipynb` | Mean of each production/welfare trait per integer THI value. |
| `Climate_Signal_Validation.ipynb` | Data-confidence checks on the climate signal. |

## `clustering/` — population structure
| Notebook | What it does |
|---|---|
| `Clustering.ipynb` | Original clustering (K-Means / hierarchical, incl. PCA). **Presented to supervisors.** |
| `Clustering_TimeSeries.ipynb` | Time-series clustering. **Presented to supervisors.** |
| `Clustering_FeaturePairs.ipynb` | Two-real-features-at-a-time clustering (post-PCA feedback). |
| `Clustering_ParityStratified.ipynb` | DBSCAN continuum check stratified by parity. |
| `Clustering_CurveShape_Heat.ipynb` | Wood's lactation-curve-shape clusters vs heat. |
| `FeaturePairs_Season_Climate.ipynb` | Feature pairs split by season / climate. |

## `forecasting/` — hierarchical milk forecasting (secondary, off-proposal)
| Notebook | What it does |
|---|---|
| `HTS.ipynb` | Hierarchical structure / overview for Total→Farm→Animal milk series. |
| `HTS_Forecasting.ipynb` | Reconciled hierarchical forecasts + evaluation. |
| `Hierarchical.ipynb` | Wood's-curve interpolation and milk/protein tensors. |
| `Hierchical_Structure.ipynb` | Builds the long HTS panel / series lists. |
