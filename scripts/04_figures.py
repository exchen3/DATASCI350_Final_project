import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# paths
u5_path = Path("data/clean/u5_mort_clean.csv")
adol_path = Path("clean/adol_fert_clean.csv")
life_path = Path("data/clean/life_expectancy_1990_2023_clean.csv")
gdp_path = Path("data/clean/gdp_cleaned.csv")

fig_dir = Path("figures")
table_dir = Path("outputs/tables")

fig_dir.mkdir(exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)

# load data
u5 = pd.read_csv(u5_path)
adol = pd.read_csv(adol_path)
life = pd.read_csv(life_path)
gdp = pd.read_csv(gdp_path)

# clean GDP column names
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

# make sure numeric columns are numeric
for df in [u5, adol, life, gdp]:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

u5["u5_mort"] = pd.to_numeric(u5["u5_mort"], errors="coerce")
adol["adol_fert"] = pd.to_numeric(adol["adol_fert"], errors="coerce")
life["life_exp"] = pd.to_numeric(life["life_exp"], errors="coerce")
gdp["gdp_per_capita"] = pd.to_numeric(gdp["gdp_per_capita"], errors="coerce")

# restrict to research plan years
u5 = u5[(u5["year"] >= 1990) & (u5["year"] <= 2023)]
adol = adol[(adol["year"] >= 1990) & (adol["year"] <= 2023)]
life = life[(life["year"] >= 1990) & (life["year"] <= 2023)]
gdp = gdp[(gdp["year"] >= 1990) & (gdp["year"] <= 2023)]

# merge core indicators
panel = (
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

# remove observations without income group
panel = panel.dropna(subset=["income_group"])

# remove non-country aggregates if they appear without a normal income group label
panel = panel[panel["income_group"].isin([
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "High income"
])]

panel = panel.sort_values(["country_code", "year"]).reset_index(drop=True)

# create 5-year lag by matching actual year, not just previous row
u5_lag = panel[["country_code", "year", "u5_mort"]].copy()
u5_lag["year"] = u5_lag["year"] + 5
u5_lag = u5_lag.rename(columns={"u5_mort": "u5_mort_lag5"})

panel = panel.merge(
    u5_lag,
    on=["country_code", "year"],
    how="left"
)

# save merged panel preview for checking
panel.to_csv(table_dir / "eda_merged_panel_for_figures.csv", index=False)

# EDA summary table
eda_summary = (
    panel.groupby("income_group", as_index=False)
    .agg(
        n_countries=("country_code", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_observations=("country_code", "count"),
        avg_u5_mort=("u5_mort", "mean"),
        avg_adol_fert=("adol_fert", "mean"),
        avg_life_exp=("life_exp", "mean"),
        avg_gdp_per_capita=("gdp_per_capita", "mean")
    )
)

eda_summary.to_csv(table_dir / "eda_summary_by_income.csv", index=False)

# missingness table
missingness = pd.DataFrame({
    "variable": ["u5_mort", "adol_fert", "life_exp", "gdp_per_capita", "u5_mort_lag5"],
    "missing_count": [
        panel["u5_mort"].isna().sum(),
        panel["adol_fert"].isna().sum(),
        panel["life_exp"].isna().sum(),
        panel["gdp_per_capita"].isna().sum(),
        panel["u5_mort_lag5"].isna().sum()
    ],
    "missing_percent": [
        panel["u5_mort"].isna().mean() * 100,
        panel["adol_fert"].isna().mean() * 100,
        panel["life_exp"].isna().mean() * 100,
        panel["gdp_per_capita"].isna().mean() * 100,
        panel["u5_mort_lag5"].isna().mean() * 100
    ]
})

missingness.to_csv(table_dir / "eda_missingness_summary.csv", index=False)

# year coverage table
coverage = (
    panel.groupby("income_group", as_index=False)
    .agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_years=("year", "nunique"),
        n_countries=("country_code", "nunique")
    )
)

coverage.to_csv(table_dir / "eda_year_coverage.csv", index=False)

# plot setup
income_order = [
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "High income"
]

income_order = [x for x in income_order if x in panel["income_group"].unique()]

color_map = {
    "Low income": "#d73027",
    "Lower middle income": "#fc8d59",
    "Upper middle income": "#91bfdb",
    "High income": "#4575b4"
}

def add_fit_line(ax, x, y):
    temp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(temp) >= 2:
        coef = np.polyfit(temp["x"], temp["y"], 1)
        x_values = np.linspace(temp["x"].min(), temp["x"].max(), 100)
        y_values = coef[0] * x_values + coef[1]
        ax.plot(x_values, y_values, linestyle="--", linewidth=2, color="black")

def save_line_plot(data, value_col, title, ylabel, filename):
    plt.figure(figsize=(9, 6))

    for group in income_order:
        temp = data[data["income_group"] == group].sort_values("year")
        plt.plot(
            temp["year"],
            temp[value_col],
            marker="o",
            linewidth=2,
            label=group,
            color=color_map.get(group)
        )

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend(title="Income group")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / filename, dpi=300)
    plt.close()

# figure 1: under-5 mortality trend
u5_trend = (
    panel.groupby(["year", "income_group"], as_index=False)["u5_mort"]
    .mean()
)

save_line_plot(
    u5_trend,
    "u5_mort",
    "Under-5 Mortality Over Time by Income Group",
    "Under-5 mortality rate per 1,000 live births",
    "01_u5_trend_by_income.png"
)

# figure 2: adolescent fertility trend
adol_trend = (
    panel.groupby(["year", "income_group"], as_index=False)["adol_fert"]
    .mean()
)

save_line_plot(
    adol_trend,
    "adol_fert",
    "Adolescent Fertility Over Time by Income Group",
    "Adolescent fertility rate per 1,000 women ages 15–19",
    "02_adol_trend_by_income.png"
)

# figure 3: life expectancy trend
life_trend = (
    panel.dropna(subset=["life_exp"])
    .groupby(["year", "income_group"], as_index=False)["life_exp"]
    .mean()
)

save_line_plot(
    life_trend,
    "life_exp",
    "Life Expectancy Over Time by Income Group",
    "Life expectancy at birth (years)",
    "03_life_exp_trend_by_income.png"
)

# figure 4: GDP trend
gdp_trend = (
    panel.dropna(subset=["gdp_per_capita"])
    .groupby(["year", "income_group"], as_index=False)["gdp_per_capita"]
    .mean()
)

save_line_plot(
    gdp_trend,
    "gdp_per_capita",
    "GDP per Capita Over Time by Income Group",
    "GDP per capita, constant 2015 US$",
    "04_gdp_trend_by_income.png"
)

# figure 5: variable distributions
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

dist_vars = [
    ("u5_mort", "Under-5 mortality"),
    ("adol_fert", "Adolescent fertility"),
    ("life_exp", "Life expectancy"),
    ("gdp_per_capita", "GDP per capita")
]

axes = axes.flatten()

for ax, (var, label) in zip(axes, dist_vars):
    temp = panel[var].dropna()
    ax.hist(temp, bins=30)
    ax.set_title(label)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

fig.suptitle("Distributions of Key Variables", fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / "05_variable_distributions.png", dpi=300)
plt.close()

# figure 6: same-year scatter
scatter_df = panel.dropna(subset=["u5_mort", "adol_fert", "income_group"])

fig, ax = plt.subplots(figsize=(8, 6))

for group in income_order:
    temp = scatter_df[scatter_df["income_group"] == group]
    ax.scatter(
        temp["u5_mort"],
        temp["adol_fert"],
        alpha=0.45,
        label=group,
        color=color_map.get(group)
    )

add_fit_line(ax, scatter_df["u5_mort"], scatter_df["adol_fert"])
ax.set_title("Under-5 Mortality and Adolescent Fertility")
ax.set_xlabel("Under-5 mortality rate per 1,000 live births")
ax.set_ylabel("Adolescent fertility rate per 1,000 women ages 15–19")
ax.legend(title="Income group")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "06_u5_vs_adol_scatter.png", dpi=300)
plt.close()

# figure 7: faceted scatter by income group
n_groups = len(income_order)
n_cols = 2
n_rows = int(np.ceil(n_groups / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4.5 * n_rows), sharex=True, sharey=True)
axes = np.array(axes).reshape(-1)

for ax, group in zip(axes, income_order):
    temp = scatter_df[scatter_df["income_group"] == group]
    ax.scatter(
        temp["u5_mort"],
        temp["adol_fert"],
        alpha=0.5,
        color=color_map.get(group)
    )
    add_fit_line(ax, temp["u5_mort"], temp["adol_fert"])
    ax.set_title(group)
    ax.set_xlabel("Under-5 mortality")
    ax.set_ylabel("Adolescent fertility")
    ax.grid(alpha=0.3)

for ax in axes[n_groups:]:
    ax.set_visible(False)

fig.suptitle("Under-5 Mortality vs. Adolescent Fertility by Income Group", fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / "07_u5_vs_adol_faceted_by_income.png", dpi=300)
plt.close()

# figure 8: lagged relationship
lag_df = panel.dropna(subset=["u5_mort_lag5", "adol_fert", "income_group"])

fig, ax = plt.subplots(figsize=(8, 6))

for group in income_order:
    temp = lag_df[lag_df["income_group"] == group]
    ax.scatter(
        temp["u5_mort_lag5"],
        temp["adol_fert"],
        alpha=0.45,
        label=group,
        color=color_map.get(group)
    )

add_fit_line(ax, lag_df["u5_mort_lag5"], lag_df["adol_fert"])
ax.set_title("Five-Year Lagged Under-5 Mortality and Current Adolescent Fertility")
ax.set_xlabel("Under-5 mortality rate five years earlier")
ax.set_ylabel("Current adolescent fertility rate")
ax.legend(title="Income group")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "08_lagged_u5_vs_adol_scatter.png", dpi=300)
plt.close()

print("EDA and visualization script finished.")
print(f"Merged panel rows: {len(panel)}")
print(f"Countries: {panel['country_code'].nunique()}")
print(f"Years: {int(panel['year'].min())}–{int(panel['year'].max())}")
print("Saved EDA tables to outputs/tables/")
print("Saved figures to figures/")
