import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

# paths
u5_path = Path("data/clean/u5_mort_clean.csv")
adol_path = Path("data/clean/adol_fert_clean.csv")
life_path = Path("data/clean/life_expectancy_1990_2023_clean.csv")
gdp_path = Path("data/clean/gdp_cleaned.csv")

processed_dir = Path("data/processed")
table_dir = Path("outputs/tables")

processed_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)

# load clean data
u5 = pd.read_csv(u5_path)
adol = pd.read_csv(adol_path)
life = pd.read_csv(life_path)
gdp = pd.read_csv(gdp_path)

# standardize GDP columns
gdp = gdp.rename(
    columns={
        "Country Name": "country_name",
        "Country Code": "country_code",
        "Year": "year",
        "GDP_per_capita": "gdp_per_capita",
        "Region": "region",
        "IncomeGroup": "income_group",
        "Decade": "decade"
    }
)

# keep useful columns
u5 = u5[["country_name", "country_code", "region", "income_group", "year", "u5_mort"]].copy()
adol = adol[["country_name", "country_code", "region", "income_group", "year", "adol_fert"]].copy()
life = life[["country_name", "country_code", "year", "life_exp"]].copy()
gdp = gdp[["country_name", "country_code", "region", "income_group", "year", "gdp_per_capita"]].copy()

# numeric conversion
for data in [u5, adol, life, gdp]:
    data["year"] = pd.to_numeric(data["year"], errors="coerce")

u5["u5_mort"] = pd.to_numeric(u5["u5_mort"], errors="coerce")
adol["adol_fert"] = pd.to_numeric(adol["adol_fert"], errors="coerce")
life["life_exp"] = pd.to_numeric(life["life_exp"], errors="coerce")
gdp["gdp_per_capita"] = pd.to_numeric(gdp["gdp_per_capita"], errors="coerce")

# restrict to research-plan years
u5 = u5[(u5["year"] >= 1990) & (u5["year"] <= 2023)]
adol = adol[(adol["year"] >= 1990) & (adol["year"] <= 2023)]
life = life[(life["year"] >= 1990) & (life["year"] <= 2023)]
gdp = gdp[(gdp["year"] >= 1990) & (gdp["year"] <= 2023)]

# merge clean data
df = (
    u5.merge(
        adol[["country_code", "year", "adol_fert"]],
        on=["country_code", "year"],
        how="inner"
    )
    .merge(
        life[["country_code", "year", "life_exp"]],
        on=["country_code", "year"],
        how="left"
    )
    .merge(
        gdp[["country_code", "year", "gdp_per_capita"]],
        on=["country_code", "year"],
        how="left"
    )
)

# keep standard income groups
df = df.dropna(subset=["income_group"])
df = df[df["income_group"].isin([
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "High income"
])].copy()

df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

# create lag variables by matching actual year
for lag in [3, 5, 10]:
    lag_df = df[["country_code", "year", "u5_mort"]].copy()
    lag_df["year"] = lag_df["year"] + lag
    lag_df = lag_df.rename(columns={"u5_mort": f"u5_mort_lag{lag}"})
    df = df.merge(lag_df, on=["country_code", "year"], how="left")

# check positivity before log transform
for col in ["u5_mort", "adol_fert"]:
    nonpositive = (df[col] <= 0).sum()
    if nonpositive > 0:
        raise ValueError(f"{col} has {nonpositive} zero or negative values. Use log1p instead.")

# create log variables
df["log_u5_mort"] = np.log(df["u5_mort"])
df["log_adol_fert"] = np.log(df["adol_fert"])

# lag variables may be missing for early years, but observed lag values must be positive
for lag in [3, 5, 10]:
    lag_col = f"u5_mort_lag{lag}"
    log_col = f"log_u5_mort_lag{lag}"

    positive_lag = df[lag_col] > 0
    df[log_col] = np.nan
    df.loc[positive_lag, log_col] = np.log(df.loc[positive_lag, lag_col])

# GDP may have missing values; only log positive GDP values
df["log_gdp_per_capita"] = np.where(
    df["gdp_per_capita"] > 0,
    np.log(df["gdp_per_capita"]),
    np.nan
)

# save merged panel after log variables are created
df.to_csv(processed_dir / "wdi_log_merged_panel.csv", index=False)

# regression-ready data for main log model
log_reg_df = df.dropna(
    subset=[
        "log_adol_fert",
        "log_u5_mort_lag5",
        "life_exp",
        "income_group"
    ]
).copy()

log_reg_df.to_csv(processed_dir / "wdi_log_regression_ready.csv", index=False)

# baseline log model
model_log_baseline = smf.ols(
    "log_adol_fert ~ log_u5_mort_lag5 + life_exp + C(income_group)",
    data=log_reg_df
).fit()

with open(table_dir / "log_baseline_regression_summary.txt", "w") as f:
    f.write("LOG BASELINE REGRESSION\n")
    f.write("=" * 70 + "\n\n")
    f.write("Formula: log_adol_fert ~ log_u5_mort_lag5 + life_exp + C(income_group)\n\n")
    f.write(model_log_baseline.summary().as_text())

print("Log merge and baseline regression finished.")
print(f"Full merged rows: {len(df)}")
print(f"Log baseline regression rows: {len(log_reg_df)}")
print(f"Countries: {df['country_code'].nunique()}")
print(f"Full year range: {int(df['year'].min())}–{int(df['year'].max())}")
print("Saved:")
print(processed_dir / "wdi_log_merged_panel.csv")
print(processed_dir / "wdi_log_regression_ready.csv")
print(table_dir / "log_baseline_regression_summary.txt")
