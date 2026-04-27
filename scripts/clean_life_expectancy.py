import pandas as pd
from pathlib import Path

# file paths
raw_path = Path("data/raw/life_expectancy/API_SP.DYN.LE00.IN_DS2_en_csv_v2_163.csv")
clean_path = Path("data/clean/life_expectancy_1990_2023_clean.csv")
coverage_path = Path("data/clean/life_expectancy_coverage_summary.csv")

clean_path.parent.mkdir(parents=True, exist_ok=True)

# read raw World Bank file
df = pd.read_csv(raw_path)
df.columns = df.columns.str.strip()

# check required columns
required_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]

missing_required = [col for col in required_cols if col not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

# keep years used in the research plan
year_cols = [str(year) for year in range(1990, 2024)]

missing_years = [col for col in year_cols if col not in df.columns]
if missing_years:
    raise ValueError(f"Missing expected year columns: {missing_years}")

# reshape from wide to long format
df_wide = df[required_cols + year_cols].copy()
df_wide = df_wide.dropna(subset=year_cols, how="all")

df_long = df_wide.melt(
    id_vars=required_cols,
    value_vars=year_cols,
    var_name="year",
    value_name="life_exp"
)

# clean column names and values
df_long = df_long.rename(
    columns={
        "Country Name": "country_name",
        "Country Code": "country_code",
        "Indicator Name": "indicator_name",
        "Indicator Code": "indicator_code",
    }
)

df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce")
df_long["life_exp"] = pd.to_numeric(df_long["life_exp"], errors="coerce")

df_long = df_long.dropna(subset=["country_name", "country_code", "year", "life_exp"])
df_long["year"] = df_long["year"].astype(int)

# make sure each country-year appears once
df_long = df_long.drop_duplicates(subset=["country_code", "year"])
df_long = df_long.sort_values(["country_code", "year"]).reset_index(drop=True)

# create coverage summary for data description
coverage = (
    df_long.groupby(["country_name", "country_code"], as_index=False)
    .agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_years=("year", "nunique"),
        missing_years_1990_2023=("year", lambda x: 34 - x.nunique()),
        avg_life_exp=("life_exp", "mean")
    )
    .sort_values(["n_years", "country_name"], ascending=[False, True])
)

# save outputs
df_long.to_csv(clean_path, index=False)
coverage.to_csv(coverage_path, index=False)

print("Life expectancy cleaning complete.")
print(f"Saved cleaned data to: {clean_path}")
print(f"Saved coverage summary to: {coverage_path}")
print(f"Rows: {len(df_long)}")
print(f"Countries / economies: {df_long['country_code'].nunique()}")
print(f"Years: {df_long['year'].min()}–{df_long['year'].max()}")
print()
print("Preview:")
print(df_long.head())
