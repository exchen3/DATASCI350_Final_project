import pandas as pd
import statsmodels.api as sm
import os

# -----------------------------
# 0. Create folders
# -----------------------------
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# 1. Load data
# -----------------------------
u5 = pd.read_csv("data/clean/u5_mort_clean.csv")
life = pd.read_csv("data/clean/life_expectancy_2015_2024_clean.csv")
gdp = pd.read_csv("data/clean/gdp_cleaned.csv")

print("Loaded shapes:")
print("u5:", u5.shape)
print("life:", life.shape)
print("gdp:", gdp.shape)

# -----------------------------
# 2. Fix LIFE (wide → long)
# -----------------------------
life = life.rename(columns={"Country Code": "country_code"})

life_long = life.melt(
    id_vars=["country_code"],
    value_vars=[str(y) for y in range(2015, 2025)],
    var_name="year",
    value_name="life_exp"
)

life_long["year"] = life_long["year"].astype(int)

# -----------------------------
# 3. Fix GDP
# -----------------------------
gdp = gdp.rename(columns={
    "Country Code": "country_code",
    "Year": "year",
    "GDP_per_capita": "gdp"
})

# -----------------------------
# 4. Merge datasets
# -----------------------------
df = u5.merge(life_long, on=["country_code", "year"], how="inner")
df = df.merge(
    gdp[["country_code", "year", "gdp"]],
    on=["country_code", "year"],
    how="left"
)

print("Merged shape:", df.shape)

# Save merged dataset
df.to_csv("data/processed/wdi_merged_panel.csv", index=False)

# -----------------------------
# 5. Basic cleaning
# -----------------------------
df = df.sort_values(["country_code", "year"])

# -----------------------------
# 6. Create lag variables
# -----------------------------
df["u5_mort_lag3"] = df.groupby("country_code")["u5_mort"].shift(3)
df["u5_mort_lag5"] = df.groupby("country_code")["u5_mort"].shift(5)
df["u5_mort_lag10"] = df.groupby("country_code")["u5_mort"].shift(10)

print("Lag variables created")

# -----------------------------
# 7. Regression function
# -----------------------------
def run_model(data, lag_var, controls):
    temp = data[[lag_var, "life_exp"] + controls].dropna()
    
    n_predictors = 1 + len(controls) + 1  # lag var + controls + constant
    if temp.shape[0] <= n_predictors:
        print(f"⚠️ Not enough data for {lag_var} (only {temp.shape[0]} rows)")
        return None
    
    X = temp[[lag_var] + controls]
    X = sm.add_constant(X)
    y = temp["life_exp"]
    
    model = sm.OLS(y, X).fit()
    print(f"✓ {lag_var}: fitted on {temp.shape[0]} rows")
    return model

# -----------------------------
# 8. Run models
# -----------------------------
model_baseline = run_model(df, "u5_mort_lag5", [])
model_lag3 = run_model(df, "u5_mort_lag3", [])
model_lag10 = run_model(df, "u5_mort_lag10", [])

if "gdp" in df.columns:
    model_gdp = run_model(df, "u5_mort_lag5", ["gdp"])
else:
    model_gdp = None

# -----------------------------
# 9. Save results
# -----------------------------
with open("outputs/tables/robustness_results.txt", "w") as f:
    
    if model_baseline is not None:
        f.write("=== Baseline (lag5) ===\n")
        f.write(model_baseline.summary().as_text())

    if model_lag3 is not None:
        f.write("\n\n=== Lag 3 ===\n")
        f.write(model_lag3.summary().as_text())

    if model_lag10 is not None:
        f.write("\n\n=== Lag 10 ===\n")
        f.write(model_lag10.summary().as_text())

    if model_gdp is not None:
        f.write("\n\n=== With GDP Control ===\n")
        f.write(model_gdp.summary().as_text())

print("✅ DONE. Results saved to outputs/tables/")

import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

plt.scatter(df["u5_mort"], df["life_exp"], alpha=0.3)
plt.xlabel("Under-5 Mortality")
plt.ylabel("Life Expectancy")
plt.title("Life Expectancy vs Under-5 Mortality")

plt.savefig("outputs/figures/scatter_life_vs_mortality.png")
plt.close()

print("📊 Figure saved to outputs/figures/")