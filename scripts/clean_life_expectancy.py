import pandas as pd
from pathlib import Path

# Define file paths
raw_path = Path("data/raw/life_expectancy/API_SP.DYN.LE00.IN_DS2_en_csv_v2_163.csv")
clean_path = Path("data/clean/life_expectancy_2015_2024_clean.csv")

# Read the raw World Bank CSV
# This file already starts with the column header, so we do NOT use skiprows=4.
df = pd.read_csv(raw_path)

# Clean column names just in case there are extra spaces
df.columns = df.columns.str.strip()

# Drop trailing empty column and all-null 2025 column if they exist
cols_to_drop = [col for col in ["Unnamed: 70", "2025"] if col in df.columns]
df = df.drop(columns=cols_to_drop)

# Keep only last 10 years + metadata columns
meta_cols = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
year_cols = [str(y) for y in range(2015, 2025)]

df_clean = df[meta_cols + year_cols].copy()

# Drop rows where all year values are missing
df_clean = df_clean.dropna(subset=year_cols, how="all")

# Drop rows with missing Country Name or Country Code
df_clean = df_clean.dropna(subset=["Country Name", "Country Code"])

# Save cleaned dataset
clean_path.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(clean_path, index=False)

print(f"Done: {df_clean.shape[0]} rows × {df_clean.shape[1]} cols")
print(df_clean.head())
