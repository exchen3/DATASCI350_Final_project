import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# paths
panel_path = Path("data/processed/wdi_log_merged_panel.csv")
table_dir = Path("outputs/tables")
fig_dir = Path("figures")
doc_dir = Path("documentation")

table_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(exist_ok=True)
doc_dir.mkdir(exist_ok=True)

# load log panel
df = pd.read_csv(panel_path)

# EDA summary for log workflow
log_eda_summary = (
    df.groupby("income_group", as_index=False)
    .agg(
        n_countries=("country_code", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_observations=("country_code", "count"),
        avg_log_u5_mort=("log_u5_mort", "mean"),
        avg_log_adol_fert=("log_adol_fert", "mean"),
        avg_life_exp=("life_exp", "mean"),
        avg_log_gdp_per_capita=("log_gdp_per_capita", "mean")
    )
)

log_eda_summary.to_csv(table_dir / "log_eda_summary_by_income.csv", index=False)

# missingness summary
missingness = pd.DataFrame({
    "variable": [
        "log_u5_mort",
        "log_u5_mort_lag3",
        "log_u5_mort_lag5",
        "log_u5_mort_lag10",
        "log_adol_fert",
        "life_exp",
        "log_gdp_per_capita"
    ],
    "missing_count": [
        df["log_u5_mort"].isna().sum(),
        df["log_u5_mort_lag3"].isna().sum(),
        df["log_u5_mort_lag5"].isna().sum(),
        df["log_u5_mort_lag10"].isna().sum(),
        df["log_adol_fert"].isna().sum(),
        df["life_exp"].isna().sum(),
        df["log_gdp_per_capita"].isna().sum()
    ],
    "missing_percent": [
        df["log_u5_mort"].isna().mean() * 100,
        df["log_u5_mort_lag3"].isna().mean() * 100,
        df["log_u5_mort_lag5"].isna().mean() * 100,
        df["log_u5_mort_lag10"].isna().mean() * 100,
        df["log_adol_fert"].isna().mean() * 100,
        df["life_exp"].isna().mean() * 100,
        df["log_gdp_per_capita"].isna().mean() * 100
    ]
})

missingness.to_csv(table_dir / "log_missingness_summary.csv", index=False)

# plot setup
income_order = [
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "High income"
]
income_order = [x for x in income_order if x in df["income_group"].unique()]

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

# figure 9: same-year log relationship
plot_df = df.dropna(subset=["log_u5_mort", "log_adol_fert", "income_group"])

fig, ax = plt.subplots(figsize=(8, 6))

for group in income_order:
    temp = plot_df[plot_df["income_group"] == group]
    ax.scatter(
        temp["log_u5_mort"],
        temp["log_adol_fert"],
        alpha=0.45,
        label=group,
        color=color_map.get(group)
    )

add_fit_line(ax, plot_df["log_u5_mort"], plot_df["log_adol_fert"])

ax.set_title("Log Under-5 Mortality and Log Adolescent Fertility")
ax.set_xlabel("Log under-5 mortality rate")
ax.set_ylabel("Log adolescent fertility rate")
ax.legend(title="Income group")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "09_log_u5_vs_adol_scatter.png", dpi=300)
plt.close()

# figure 10: lagged log relationship
lag_plot_df = df.dropna(subset=["log_u5_mort_lag5", "log_adol_fert", "income_group"])

fig, ax = plt.subplots(figsize=(8, 6))

for group in income_order:
    temp = lag_plot_df[lag_plot_df["income_group"] == group]
    ax.scatter(
        temp["log_u5_mort_lag5"],
        temp["log_adol_fert"],
        alpha=0.45,
        label=group,
        color=color_map.get(group)
    )

add_fit_line(ax, lag_plot_df["log_u5_mort_lag5"], lag_plot_df["log_adol_fert"])

ax.set_title("Log Five-Year Lagged Under-5 Mortality and Log Adolescent Fertility")
ax.set_xlabel("Log under-5 mortality rate five years earlier")
ax.set_ylabel("Log current adolescent fertility rate")
ax.legend(title="Income group")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "10_log_lagged_u5_vs_adol_scatter.png", dpi=300)
plt.close()

# documentation note
with open(doc_dir / "log_analysis_notes.md", "w") as f:
    f.write("# Log Analysis Notes\n\n")
    f.write("This file documents the parallel log-transformed analysis.\n\n")
    f.write("The original workflow remains unchanged. This additional workflow creates log-transformed versions of under-5 mortality and adolescent fertility to check whether the relationship is robust to skewness.\n\n")
    f.write("Main log model:\n\n")
    f.write("`log_adol_fert ~ log_u5_mort_lag5 + life_exp + C(income_group)`\n\n")
    f.write("Additional checks include lag 3, lag 10, and a GDP-control model.\n\n")
    f.write("Because all mortality and fertility values are positive in the regression-ready sample, the workflow uses natural log transformation with `np.log()` instead of `np.log1p()`.\n")

print("Log figures and EDA tables finished.")
print("Saved:")
print(table_dir / "log_eda_summary_by_income.csv")
print(table_dir / "log_missingness_summary.csv")
print(fig_dir / "09_log_u5_vs_adol_scatter.png")
print(fig_dir / "10_log_lagged_u5_vs_adol_scatter.png")
print(doc_dir / "log_analysis_notes.md")
