# %% [markdown]
# # Stage 1 — Data Foundation & Integrity
#
# Clean-rebuild pipeline, ANALYSIS-ONLY pass. This stage produces one audited,
# reproducible analysis table and proves the production data + weather merge are
# trustworthy before any inference (see `ANALYSIS_PIPELINE_SPEC.md`, Stage 1).
#
# **What it does**
# 1. Load the cleaned production table (`After_Processing.csv`, output of `FeatureEng_v2`).
# 2. Schema / counts / duplicate audit; provenance cross-check vs raw `Dati_ok.txt`.
# 3. Re-derive SCS / DIM / AFC / ECM from their inputs and confirm they match the stored columns.
# 4. Range, plausibility, and missingness audit.
# 5. Build analysis variables: SOL class, parity class, season (2 defs, D2), macro-area.
# 6. Rebuild the climate layer from the per-farm daily Open-Meteo files (D1):
#    farm-month THI normal -> daily THI anomaly (R1); merge same-day weather onto test-days.
# 7. Cross-check the fresh merge against the old `Final_Merged_Data.csv`.
# 8. Persist `analysis_base.parquet` + `farm_daily_anom.parquet` + a data dictionary.
#
# Correctness rules enforced here: R1 (anomaly = THI - farm-month normal), R7 (THI_1 is
# the Matera index), R8 (seed/versions/data-hash logged). No rows are dropped silently:
# every filter reports its count.

# %%
import pandas as pd, numpy as np, hashlib, glob, os, sys, json, platform
from pathlib import Path

RNG_SEED = 42
np.random.seed(RNG_SEED)

# --- robust path resolution (works from repo root or from codes/pipeline/) ---
def find_root():
    cands = [Path("Thesis_Data"), Path("../../Thesis_Data")]
    try:
        cands.append(Path(__file__).resolve().parents[2] / "Thesis_Data")   # script run
    except NameError:
        pass                                                                # notebook run
    for cand in cands:
        if cand.exists():
            return cand.resolve().parent
    raise SystemExit("Cannot locate Thesis_Data")
ROOT = find_root()
DATA = ROOT / "Thesis_Data"
FINAL = DATA / "Final_Data"
WEATHER_DIR = DATA / "farm_weather_indices"
OUT = ROOT / "codes" / "pipeline" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def rule(msg): print("\n" + "="*78 + f"\n{msg}\n" + "="*78)
def check(name, ok, detail=""):
    tag = "PASS " if ok else "FAIL "
    print(f"  [{tag}] {name}" + (f"  --  {detail}" if detail else ""))
    return ok

rule("ENVIRONMENT")
print(f"  python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__} | seed {RNG_SEED}")

# %% [markdown]
# ## 1.1 Load cleaned production table + provenance cross-check vs raw

# %%
rule("1.1  LOAD & PROVENANCE")
ap_path = FINAL / "After_Processing.csv"
# log a data-hash (R8)
h = hashlib.md5()
with open(ap_path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(f"  After_Processing.csv md5 = {h.hexdigest()}")

df = pd.read_csv(ap_path)
for c in ["dtb", "dtc", "dtt"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
# AFC is stored as a timedelta string -> integer days
df["AFC_days"] = pd.to_timedelta(df["AFC"], errors="coerce").dt.days
print(f"  rows={len(df):,} | animals={df.Animal_ID.nunique():,} | farms={df.Farm_Code.nunique():,}")
print(f"  test-day range: {df.dtt.min().date()} -> {df.dtt.max().date()}")

# raw provenance (row count only; After_Processing is the edited subset)
raw_n = sum(1 for _ in open(DATA / "Dati_ok.txt")) - 1
print(f"  raw Dati_ok.txt rows={raw_n:,}  ->  kept {len(df)/raw_n:.1%} after FeatureEng_v2 editing")

# %% [markdown]
# ## 1.2 Schema, uniqueness, NM

# %%
rule("1.2  SCHEMA / UNIQUENESS + EDITING")
# --- duplicate (Animal_ID, dtt) rows: exact-duplicate injections (likely the region/latlong
#     merge). Drop exact dups, then resolve rare same-key collisions preferring the more
#     complete row (non-null dtb). This is the only place rows are removed for dedup. ---
n0 = len(df)
dup0 = df.duplicated(subset=["Animal_ID","dtt"]).sum()
print(f"  duplicate (Animal_ID,dtt) rows before dedup: {dup0:,}")
df = df.sort_values("dtb", na_position="last")
df = df.drop_duplicates()                                            # fully-identical rows
df = df.drop_duplicates(subset=["Animal_ID","dtt"], keep="first")    # rare differing-value collisions
df = df.reset_index(drop=True)
print(f"  dropped {n0-len(df):,} duplicate rows ({(n0-len(df))/n0:.2%}) -> {len(df):,} rows")
check("no duplicate (Animal_ID, dtt) keys after dedup",
      df.duplicated(subset=["Animal_ID","dtt"]).sum()==0)

# --- DIM bound (Costa 2020: 5-400); editing left a handful outside ---
n1 = len(df)
df = df[df.DIM.between(5,400)].reset_index(drop=True)
print(f"  dropped {n1-len(df):,} rows outside DIM[5,400] -> {len(df):,} rows")

check("dates parsed (no NaT in dtt/dtc)", df.dtt.isna().sum()==0 and df.dtc.isna().sum()==0,
      f"dtt NaT={df.dtt.isna().sum()}, dtc NaT={df.dtc.isna().sum()}")
nm = df.NM.value_counts(dropna=False).to_dict()
check("NM == 2 only (editing applied)", set(df.NM.unique()) == {2}, f"NM values={nm}")
# parity has a smooth biological tail to ~13 (old buffaloes) -> keep records, cap only the class
print(f"  parity range {df.parity.min()}-{df.parity.max()} "
      f"({(df.parity>6).mean():.1%} > 6, real; capped to 5+ in parity_class for models)")

# %% [markdown]
# ## 1.3 Derived-variable identity checks (recompute from inputs, compare to stored)
# SCS, DIM, AFC, ECM must equal their formulas exactly (or within float tol).

# %%
rule("1.3  DERIVED-VARIABLE IDENTITIES")
# SCS = log2(cells/100) + 3   (Ali & Shook 1980)
m = df.cells > 0
scs_calc = np.log2(df.loc[m, "cells"]/100) + 3
scs_err = (scs_calc - df.loc[m, "SCS"]).abs().max()
check("SCS == log2(cells/100)+3", scs_err < 1e-6, f"max|Δ|={scs_err:.2e}; cells<=0: {(~m).sum()}")

# DIM = dtt - dtc
dim_calc = (df.dtt - df.dtc).dt.days
dim_err = (dim_calc - df.DIM).abs().max()
check("DIM == (dtt - dtc) days", dim_err == 0, f"max|Δ|={dim_err}")

# AFC = dtc - dtb (age at that calving, as stored)
afc_calc = (df.dtc - df.dtb).dt.days
afc_err = (afc_calc - df.AFC_days).abs().max()
check("AFC == (dtc - dtb) days", afc_err == 0, f"max|Δ|={afc_err}")

# ECM: FeatureEng formula vs stored, and vs Sjaunja(1990) reference
ecm_feateng = ((((df.fat_p*10) - 40) + ((df.protein_p*10) - 31)) * 0.01155 + 1) * df.milk_kg
ecm_err = (ecm_feateng - df.ECM).abs().max()
check("ECM reproduces FeatureEng formula", ecm_err < 1e-3, f"max|Δ|={ecm_err:.2e}")
ecm_sjaunja = df.milk_kg * (0.25 + 0.122*df.fat_p + 0.077*df.protein_p)   # Sjaunja et al. 1990
ratio = (df.ECM / df.milk_kg)
print(f"  ECM/milk ratio: mean={ratio.mean():.3f} (buffalo >1 expected: fat~{df.fat_p.mean():.1f}%, prot~{df.protein_p.mean():.1f}%)")
print(f"  FeatureEng-ECM vs Sjaunja-1990-ECM: corr={ecm_feateng.corr(ecm_sjaunja):.4f}, "
      f"mean ratio={ (ecm_feateng/ecm_sjaunja).mean():.3f}  -> D3: formula is a standard g/kg energy correction")

# %% [markdown]
# ## 1.4 Ranges, plausibility, missingness

# %%
rule("1.4  RANGES / PLAUSIBILITY / MISSINGNESS")
desc = df[["milk_kg","fat_p","protein_p","cells","SCS","ECM","DIM","parity","AFC_days"]].describe(
        percentiles=[.01,.5,.99]).T[["min","1%","50%","99%","max","mean"]]
print(desc.round(2).to_string())
plaus = {
    "milk_kg in [0.4,26.5]": df.milk_kg.between(0.4,26.5).mean(),
    "fat_p in [1.6,15.05]":  df.fat_p.between(1.6,15.05).mean(),
    "protein_p in [2,7]":    df.protein_p.between(2,7).mean(),
    "DIM in [5,400]":        df.DIM.between(5,400).mean(),
    "parity >= 1 (biological tail kept)": (df.parity>=1).mean(),
}
print()
for k,v in plaus.items():
    check(k, v > 0.999, f"{v:.4%} in range")
miss = df.isna().mean().sort_values(ascending=False)
miss = miss[miss>0]
print("\n  columns with missingness:")
print("   ", (miss.round(4).to_dict() if len(miss) else "none"))

# %% [markdown]
# ## 1.5 Analysis variables: SOL class, parity class, season (2 defs, D2), macro-area

# %%
rule("1.5  ANALYSIS VARIABLES")
df["year"]  = df.dtt.dt.year
df["month"] = df.dtt.dt.month
df["dim30"] = np.minimum(np.ceil(df.DIM/30).astype(int), 12)          # 12 x 30-day SOL classes
df["parity_class"] = np.minimum(df.parity, 5)                          # 1,2,3,4,5+
# TRUE age at first calving = (first calving date for the animal) - birth. The stored AFC is
# dtc-dtb at the CURRENT calving (grows with parity, redundant); true_AFC is constant per animal.
first_calv = df.groupby("Animal_ID")["dtc"].transform("min")
df["true_AFC_days"] = (first_calv - df.dtb).dt.days
print(f"  true_AFC (age at 1st calving): median={df.true_AFC_days.median():.0f} d "
      f"(~{df.true_AFC_days.median()/30.44:.1f} mo) vs stored AFC median "
      f"{df.AFC_days.median():.0f} d -- stored AFC != age at first calving")
# season (D2 - build BOTH, compare later)
df["is_summer_warmhalf"] = df.month.isin([4,5,6,7,8,9])                # Apr-Sep
df["is_summer_strict"]   = df.month.isin([6,7,8,9])                    # Jun-Sep
season4 = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
           6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}
df["season4"] = df.month.map(season4)

# macro-area from Region (ISTAT) - keyword match to survive spelling variants
NORTH   = ["piemonte","valle","aosta","lombardia","liguria","trentino","alto adige","veneto","friuli","emilia"]
CENTRAL = ["toscana","umbria","marche","lazio"]
SOUTH   = ["abruzzo","molise","campania","puglia","basilicata","calabria","sicil","sardegn"]
def macro(r):
    s = str(r).strip().lower()
    if any(k in s for k in NORTH):   return "North"
    if any(k in s for k in CENTRAL): return "Central"
    if any(k in s for k in SOUTH):   return "South"
    return "UNMAPPED"
df["macro_area"] = df.Region.map(macro)
unmapped = sorted(df.loc[df.macro_area=="UNMAPPED","Region"].dropna().unique().tolist())
check("all Regions mapped to a macro-area", len(unmapped)==0, f"unmapped={unmapped}")
print("  macro-area record share:", (df.macro_area.value_counts(normalize=True).round(3)).to_dict())
print("  regions present:", sorted(df.Region.dropna().unique().tolist()))

# %% [markdown]
# ## 1.6 Climate layer rebuilt from daily Open-Meteo files (D1) + THI anomaly (R1)
# Farm-month normal is computed from the FULL daily series (every day), which is the
# correct climatological baseline - not just the sampled test-days.

# %%
rule("1.6  CLIMATE LAYER + THI ANOMALY")
wfiles = sorted(glob.glob(str(WEATHER_DIR / "*_with_indices.csv")))
print(f"  daily weather files: {len(wfiles)}")
wcols = ["date","temperature_avg","humidity_avg","THI_1_avg","THI_1_max"]
parts = []
for wf in wfiles:
    code = int(float(os.path.basename(wf).split("_")[1]))   # farm_<code>.0_with_indices.csv
    w = pd.read_csv(wf, usecols=wcols)
    w["Farm_Code"] = code
    parts.append(w)
daily = pd.concat(parts, ignore_index=True)
daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
daily["month"] = daily.date.dt.month
print(f"  daily rows={len(daily):,} | farms in weather={daily.Farm_Code.nunique()}")

# farm-month THI normal + daily anomaly
norm = daily.groupby(["Farm_Code","month"])["THI_1_avg"].transform("mean")
daily["THI_1_farmmonth_normal"] = norm
daily["THI_anom"] = daily["THI_1_avg"] - norm
# sanity: anomaly ~ mean 0 within farm-month
anom_mean = daily.groupby(["Farm_Code","month"])["THI_anom"].mean().abs().max()
check("THI anomaly de-means within farm-month", anom_mean < 1e-9, f"max|mean|={anom_mean:.2e}")
# seasonal cycle sanity (R7): THI_1_max should peak Jun-Aug
mo = daily.groupby("month")["THI_1_max"].mean()
check("THI_1_max peaks Jun-Aug, troughs Dec-Feb",
      mo.idxmax() in (6,7,8) and mo.idxmin() in (12,1,2),
      f"peak month={mo.idxmax()} ({mo.max():.1f}), trough={mo.idxmin()} ({mo.min():.1f})")

# farm coverage for the production animals
prod_farms = set(df.Farm_Code.unique()); wx_farms = set(daily.Farm_Code.unique())
missing_farms = prod_farms - wx_farms
check("every production farm has weather", len(missing_farms)==0, f"missing={len(missing_farms)}")

daily_out = daily[["Farm_Code","date","temperature_avg","humidity_avg","THI_1_avg","THI_1_max",
                   "THI_1_farmmonth_normal","THI_anom"]].copy()
daily_out.to_parquet(OUT / "farm_daily_anom.parquet", index=False)
print(f"  wrote {OUT/'farm_daily_anom.parquet'}  ({len(daily_out):,} rows)")

# %% [markdown]
# ## 1.7 Merge same-day weather + anomaly onto test-days; audit vs old merge

# %%
rule("1.7  TEST-DAY MERGE + AUDIT")
key_daily = daily[["Farm_Code","date","temperature_avg","humidity_avg","THI_1_avg","THI_1_max","THI_anom"]]
merged = df.merge(key_daily, left_on=["Farm_Code","dtt"], right_on=["Farm_Code","date"], how="left")
match = merged["THI_1_avg"].notna().mean()
check("test-days matched to a weather day", match > 0.999, f"{match:.4%} matched")

# cross-check against previously-validated Final_Merged_Data (light: 4 cols)
fmd = pd.read_csv(FINAL / "Final_Merged_Data.csv", usecols=["Farm_Code","Animal_ID","dtt","THI_1_avg"])
fmd["dtt"] = pd.to_datetime(fmd["dtt"], errors="coerce")
cc = merged[["Farm_Code","Animal_ID","dtt","THI_1_avg"]].merge(
        fmd, on=["Farm_Code","Animal_ID","dtt"], suffixes=("_new","_old"))
d = (cc.THI_1_avg_new - cc.THI_1_avg_old).abs()
check("fresh merge == old Final_Merged_Data THI_1_avg", d.max() < 1e-3,
      f"n={len(cc):,} max|Δ|={d.max():.2e}")

merged = merged.drop(columns=["date"])

# %% [markdown]
# ## 1.8 Persist analysis_base + data dictionary

# %%
rule("1.8  PERSIST OUTPUTS")
base_cols = ["Farm_Code","Animal_ID","dtb","dtc","dtt","year","month","season4",
             "is_summer_warmhalf","is_summer_strict","macro_area","Region","Province","City",
             "parity","parity_class","DIM","dim30","AFC_days","true_AFC_days",
             "milk_kg","fat_p","protein_p","cells","SCS","ECM",
             "milk_kg_lag1","milk_kg_lag2","protein_p_lag1","protein_p_lag2",
             "temperature_avg","humidity_avg","THI_1_avg","THI_1_max","THI_anom"]
base = merged[base_cols].copy()
base.to_parquet(OUT / "analysis_base.parquet", index=False)
print(f"  wrote {OUT/'analysis_base.parquet'}  ({len(base):,} rows x {base.shape[1]} cols)")

data_dict = {
    "rows": int(len(base)), "animals": int(base.Animal_ID.nunique()),
    "farms": int(base.Farm_Code.nunique()),
    "date_range": [str(base.dtt.min().date()), str(base.dtt.max().date())],
    "seed": RNG_SEED, "source_md5": h.hexdigest(),
    "columns": {c: str(base[c].dtype) for c in base.columns},
    "notes": {
        "THI_anom": "THI_1_avg - farm-month mean (R1 causal heat measure)",
        "season": "is_summer_warmhalf=Apr-Sep (D2a); is_summer_strict=Jun-Sep (D2b); season4=Matera calving season",
        "ECM": "FeatureEng g/kg energy correction ~= Sjaunja 1990; ECM>milk expected for buffalo",
        "weather": "Open-Meteo/ERA5 (D1), rebuilt from per-farm daily files",
    },
}
(OUT / "data_dictionary.json").write_text(json.dumps(data_dict, indent=2))
print(f"  wrote {OUT/'data_dictionary.json'}")
rule("STAGE 1 COMPLETE")
print("  outputs in codes/pipeline/outputs/: analysis_base.parquet, farm_daily_anom.parquet, data_dictionary.json")

# %% [markdown]
# ## 1.9 Figures — data foundation at a glance

# %%
rule("1.9  FIGURES")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIGD = OUT / "figures"; FIGD.mkdir(parents=True, exist_ok=True)

# Fig 1a: sample composition (records per year, per macro-area, per parity)
fig,ax=plt.subplots(1,3,figsize=(15,4))
base.groupby("year").size().plot(kind="bar",ax=ax[0],color="steelblue"); ax[0].set_title("Records per year"); ax[0].set_xlabel("")
base.macro_area.value_counts().plot(kind="bar",ax=ax[1],color="seagreen"); ax[1].set_title("Records per macro-area"); ax[1].set_xlabel("")
base.parity_class.value_counts().sort_index().plot(kind="bar",ax=ax[2],color="indianred"); ax[2].set_title("Records per parity class"); ax[2].set_xlabel("parity")
for a in ax: a.grid(alpha=.3,axis="y")
plt.tight_layout(); plt.savefig(FIGD/"s1_sample_composition.png",dpi=130); plt.show()

# Fig 1b: trait distributions (plausibility visual)
fig,ax=plt.subplots(2,3,figsize=(15,7))
for a,c,col in zip(ax.flat,["milk_kg","fat_p","protein_p","SCS","DIM","AFC_days"],
                   ["steelblue","orange","green","firebrick","slateblue","teal"]):
    a.hist(base[c].dropna(),bins=60,color=col); a.set_title(c); a.grid(alpha=.3)
plt.suptitle("Trait distributions after editing",y=1.02); plt.tight_layout(); plt.savefig(FIGD/"s1_trait_distributions.png",dpi=130); plt.show()

# Fig 1c: weather-merge audit — THI seasonal cycle (monthly)
mo = daily_out.assign(month=daily_out.date.dt.month).groupby("month")[["THI_1_avg","THI_1_max"]].mean()
plt.figure(figsize=(8,4.5))
plt.plot(mo.index,mo.THI_1_avg,marker="o",label="THI_1_avg")
plt.plot(mo.index,mo.THI_1_max,marker="s",label="THI_1_max")
plt.axhspan(72,90,color="red",alpha=.06,label="THI>72 (heat stress)")
plt.xticks(range(1,13)); plt.xlabel("month"); plt.ylabel("THI"); plt.title("Weather-merge audit: THI seasonal cycle (peaks Aug)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIGD/"s1_thi_seasonal_cycle.png",dpi=130); plt.show()
print("  wrote s1_sample_composition / s1_trait_distributions / s1_thi_seasonal_cycle")
