import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt

# -----------------------------
# 0. Create folders
# -----------------------------
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# 1. Load data
# -----------------------------
u5 = pd.read_csv("data/clean/u5_mort_clean.csv")
life = pd.read_csv("data/clean/life_expectancy_2015_2024_clean.csv")
gdp = pd.read_csv("data/clean/gdp_cleaned.csv")
fert = pd.read_csv("data/clean/adol_fert_clean.csv")

print("Loaded shapes:")
print("u5:", u5.shape)
print("life:", life.shape)
print("gdp:", gdp.shape)
print("fert:", fert.shape)

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
# 4. Fix FERTILITY
# -----------------------------
fert.columns = [c.lower() for c in fert.columns]
if "value" in fert.columns:
    fert = fert.rename(columns={"value": "adol_fert"})

# -----------------------------
# 5. Merge datasets
# -----------------------------
df = u5.merge(life_long, on=["country_code", "year"], how="inner")
df = df.merge(gdp[["country_code", "year", "gdp"]], on=["country_code", "year"], how="left")
df = df.merge(fert[["country_code", "year", "adol_fert"]], on=["country_code", "year"], how="inner")

print("Merged shape:", df.shape)
print("Columns:", df.columns.tolist())

df.to_csv("data/processed/wdi_merged_panel.csv", index=False)

# -----------------------------
# 6. Clean numeric
# -----------------------------
for col in ["u5_mort", "life_exp", "adol_fert", "gdp"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# 7. Sort & create lag
# -----------------------------
df = df.sort_values(["country_code", "year"])
df["u5_mort_lag3"]  = df.groupby("country_code")["u5_mort"].shift(3)
df["u5_mort_lag5"]  = df.groupby("country_code")["u5_mort"].shift(5)
df["u5_mort_lag10"] = df.groupby("country_code")["u5_mort"].shift(10)
print("Lag variables created")

# -----------------------------
# 8. Regression function
# -----------------------------
def run_model(data, lag_var, controls, label=""):
    # Only keep controls that actually exist in data
    available_controls = [c for c in controls if c in data.columns]
    missing = set(controls) - set(available_controls)
    if missing:
        print(f"  ℹ️ Dropping unavailable controls: {missing}")

    temp = data[[lag_var, "adol_fert"] + available_controls].dropna()

    if temp.shape[0] < 20:
        print(f"Not enough data for {lag_var} {label}")
        return None

    X = temp[[lag_var] + available_controls].copy()

    # Handle categorical income_group if present
    if "income_group" in X.columns:
        X = pd.get_dummies(X, columns=["income_group"], drop_first=True, dtype=float)

    # Safety net: force everything numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna()
    y = temp.loc[X.index, "adol_fert"]

    if X.shape[0] < 20:
        print(f"Not enough data for {lag_var} {label} after numeric cleaning")
        return None

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(f"✓ {lag_var} {label}: fitted on {X.shape[0]} rows")
    return model

# -----------------------------
# 9. Run models — Robustness suite
# -----------------------------
controls_base = ["life_exp", "income_group"]

print("\n--- Lag robustness ---")
model_baseline = run_model(df, "u5_mort_lag5", controls_base, "(baseline)")
model_lag3     = run_model(df, "u5_mort_lag3", controls_base)
model_lag10    = run_model(df, "u5_mort_lag10", controls_base)

print("\n--- GDP control ---")
model_gdp = run_model(df, "u5_mort_lag5", ["life_exp", "gdp", "income_group"], "(+GDP)")

# -----------------------------
# 10. Missingness sensitivity check
#     (only keep countries with high coverage)
# -----------------------------
print("\n--- Missingness sensitivity (high-coverage countries only) ---")
coverage = df.groupby("country_code").apply(
    lambda g: g[["u5_mort", "adol_fert", "life_exp"]].notna().all(axis=1).mean()
)
high_cov_countries = coverage[coverage >= 0.8].index
df_high_cov = df[df["country_code"].isin(high_cov_countries)].copy()
print(f"Kept {len(high_cov_countries)} of {df['country_code'].nunique()} countries with ≥80% coverage")

model_high_cov = run_model(df_high_cov, "u5_mort_lag5", controls_base, "(high-coverage subset)")

# -----------------------------
# 11. Save results
# -----------------------------
with open("outputs/tables/robustness_results.txt", "w") as f:
    f.write("ROBUSTNESS ANALYSIS RESULTS\n")
    f.write("=" * 60 + "\n")
    f.write("Outcome: adolescent fertility rate\n")
    f.write("Note: life expectancy data limited to 2015-2024,\n")
    f.write("      so lag10 is not feasible.\n\n")

    if model_baseline is not None:
        f.write("=== Baseline (lag5 + life_exp + income_group) ===\n")
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
    if model_high_cov is not None:
        f.write("\n\n=== High-Coverage Subset (≥80%) ===\n")
        f.write(model_high_cov.summary().as_text())

print("Regression results saved")

# -----------------------------
# 12. Visualization
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
plot_data = df[["u5_mort", "adol_fert"]].dropna()
ax.scatter(plot_data["u5_mort"], plot_data["adol_fert"], alpha=0.3, s=15)

x = plot_data["u5_mort"]
y = plot_data["adol_fert"]
slope, intercept = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, slope * xs + intercept, color="red", linewidth=2,
        label=f"Fit: y = {slope:.2f}x + {intercept:.1f}")

ax.set_xlabel("Under-5 Mortality (per 1,000 live births)")
ax.set_ylabel("Adolescent Fertility (births per 1,000 women aged 15-19)")
ax.set_title("Adolescent Fertility vs Under-5 Mortality")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/figures/scatter_fertility_vs_mortality.png", dpi=150)
plt.close()

print("Figure saved")