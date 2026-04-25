"""
Data integration and baseline regression script for DATASCI 350 Final Project.

Purpose:
- Merge cleaned World Bank indicators into one panel dataset.
- Create a 5-year lag of under-5 mortality.
- Run a baseline OLS regression testing whether lagged child mortality predicts adolescent fertility.

Inputs:
- data/clean/u5_mort_clean.csv
- clean/adol_fert_clean.csv or data/clean/adol_fert_clean.csv
- data/clean/life_expectancy_2015_2024_clean.csv
- data/clean/gdp_cleaned.csv if used later

Outputs:
- data/processed/wdi_merged_panel.csv
- data/processed/wdi_regression_ready.csv
- outputs/tables/baseline_regression_summary.txt

Main model:
adol_fert ~ u5_mort_lag5 + life_exp + C(income_group)
"""

from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

BASE_DIR = Path(__file__).resolve().parents[1]

CLEAN_DIRS = [
    BASE_DIR / "data" / "clean",
    BASE_DIR / "clean"
]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
TABLE_DIR = BASE_DIR / "outputs" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def find_file(filename):
    for folder in CLEAN_DIRS:
        path = folder / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename} in data/clean or clean folder.")


# 2. Load cleaned datasets
mort = pd.read_csv(find_file("u5_mort_clean.csv"))
fert = pd.read_csv(find_file("adol_fert_clean.csv"))

# Change this filename if your groupmate used a different name
life = pd.read_csv(find_file("life_expectancy_2015_2024_clean.csv"))

print("Mortality columns:", mort.columns.tolist())
print("Fertility columns:", fert.columns.tolist())
print("Life expectancy columns:", life.columns.tolist())


# Standardize column names
def standardize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


mort = standardize_columns(mort)
fert = standardize_columns(fert)
life = standardize_columns(life)

# Rename
rename_map = {
    "country_name": "country",
    "country": "country",
    "country_code": "country_code",
    "year": "year",
}

mort = mort.rename(columns=rename_map)
fert = fert.rename(columns=rename_map)
life = life.rename(columns=rename_map)

# If life expectancy file is still in wide World Bank format, convert it to long format
if "year" not in life.columns:
    life = life.rename(columns={
        "country_name": "country",
        "country_code": "country_code"
    })

    year_cols = [col for col in life.columns if col.isdigit()]

    life = life.melt(
        id_vars=["country", "country_code"],
        value_vars=year_cols,
        var_name="year",
        value_name="life_exp"
    )

    life["year"] = life["year"].astype(int)
    life["life_exp"] = pd.to_numeric(life["life_exp"], errors="coerce")

# Make sure key columns have correct type
for df in [mort, fert, life]:
    df["country_code"] = df["country_code"].astype(str)
    df["year"] = df["year"].astype(int)


# Keep useful columns
mort_cols = ["country_code", "country", "year", "u5_mort", "income_group"]
mort_cols = [col for col in mort_cols if col in mort.columns]
mort = mort[mort_cols].copy()

fert_cols = ["country_code", "country", "year", "adol_fert"]
fert_cols = [col for col in fert_cols if col in fert.columns]
fert = fert[fert_cols].copy()

life_cols = ["country_code", "country", "year", "life_exp"]
life_cols = [col for col in life_cols if col in life.columns]
life = life[life_cols].copy()


# Merge datasets
df = mort.merge(
    fert,
    on=["country_code", "year"],
    how="inner",
    suffixes=("", "_fert")
)

df = df.merge(
    life,
    on=["country_code", "year"],
    how="inner",
    suffixes=("", "_life")
)

# Clean duplicate country columns
for col in ["country_fert", "country_life"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Keep research-plan period
df = df[(df["year"] >= 1990) & (df["year"] <= 2023)].copy()

print("Merged shape:", df.shape)
print(df.head())


# Create 5-year mortality lag
df = df.sort_values(["country_code", "year"])

df["u5_mort_lag5"] = (
    df.groupby("country_code")["u5_mort"]
    .shift(5)
)

# Keep rows usable for regression
reg_df = df.dropna(subset=["adol_fert", "u5_mort_lag5", "life_exp", "income_group"]).copy()

print("Regression data shape:", reg_df.shape)


# Save merged dataset

df.to_csv(PROCESSED_DIR / "wdi_merged_panel.csv", index=False)
reg_df.to_csv(PROCESSED_DIR / "wdi_regression_ready.csv", index=False)

# Baseline regression
model = smf.ols(
    "adol_fert ~ u5_mort_lag5 + life_exp + C(income_group)",
    data=reg_df
).fit()

print(model.summary())

with open(TABLE_DIR / "baseline_regression_summary.txt", "w") as f:
    f.write(model.summary().as_text())

print("Saved:")
print(PROCESSED_DIR / "wdi_merged_panel.csv")
print(PROCESSED_DIR / "wdi_regression_ready.csv")
print(TABLE_DIR / "baseline_regression_summary.txt")
