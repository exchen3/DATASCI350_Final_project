# DATASCI 350 — Final Project
## Does Child Mortality Predict Adolescent Fertility?
### Evidence from the World Bank WDI, 1990–2023

**Authors:** Yang Lyu (2609405) · Steven Hu (2634698) · Runyan Tian ([2609567]) · Eric Chen (2607573) 
**Course:** DATASCI 350 — Data Science Computing  
**Date:** April 2026

---

## Research Question

> Does declining child mortality predict subsequent reductions in adolescent fertility, and does this relationship vary by regional income level?

---

## Repository Structure

```
DATASCI350_Final_project/
│
├── data/
│   ├── raw/                                    # Raw WDI CSVs (not tracked by git)
│   │   ├── Adol_fert/
│   │   ├── gdp/
│   │   ├── life_expectancy/
│   │   └── Mortality/
│   ├── processed/                              # Cleaned and merged datasets
│   │   ├── adol_fert_clean.csv
│   │   ├── country_coverage.csv
│   │   ├── decade_avg_by_income.csv
│   │   ├── gdp_cleaned.csv
│   │   ├── life_expectancy_1990_2014_clean.csv
│   │   ├── life_expectancy_2015_2023_clean.csv
│   │   ├── life_expectancy_coverage_summary.csv
│   │   ├── school_enrollment_clean.csv
│   │   ├── u5_mort_clean.csv
│   │   ├── wdi_merged_panel.csv
│   │   ├── wdi_regression_ready.csv
│   │   ├── wdi_log_merged_panel.csv
│   │   └── wdi_log_regression_ready.csv
│   └── clean/                                  # Final analysis-ready files
│
├── scripts/                                    # Python analysis scripts
│   ├── clean_life_expectancy.py
│   ├── clean_u5_mort.py
│   ├── 03_merge_regression.py
│   ├── 04_figures.py
│   ├── 05_log_merge_regression.py
│   ├── 06_log_robustness.py
│   ├── 07_log_figures.py
│   └── analysis_robustness.py
│
├── notebooks/                                  # Jupyter notebooks
│   ├── Mortality_data.ipynb
│   └── Final_project.ipynb
│
├── outputs/                                    # All generated outputs
│   ├── figures/
│   │   ├── 01_u5_trend_by_income.png
│   │   ├── 02_adol_trend_by_income.png
│   │   ├── 03_life_exp_trend_by_income.png
│   │   ├── 04_gdp_trend_by_income.png
│   │   ├── 05_variable_distributions.png
│   │   ├── 06_u5_vs_adol_scatter.png
│   │   ├── 07_u5_vs_adol_faceted_by_income.png
│   │   ├── 08_lagged_u5_vs_adol_scatter.png
│   │   ├── 09_log_u5_vs_adol_scatter.png
│   │   ├── 10_log_lagged_u5_vs_adol_scatter.png
│   │   ├── fig3_decade_avg_bars.png
│   │   ├── scatter_fertility_vs_mortality.png
│   │   └── scatter_life_vs_mortality.png
│   └── tables/
│       ├── baseline_regression_summary.txt
│       ├── log_baseline_regression_summary.txt
│       ├── eda_merged_panel_for_figures.csv
│       ├── eda_missingness_summary.csv
│       ├── eda_summary_by_income.csv
│       ├── eda_year_coverage.csv
│       ├── log_eda_summary_by_income.csv
│       ├── log_missingness_summary.csv
│       ├── log_model_comparison.csv
│       ├── log_robustness_results.txt
│       └── robustness_results.txt
│
├── report/                                     # Quarto report
│   ├── final_report.qmd
│   ├── final_report.pdf
│   └── final_report.html
│
├── documentation/                              # Codebook and ER diagram
│
└── README.md
```

---

## How to Run

### Prerequisites

Install required Python packages:

```bash
pip install duckdb pandas numpy matplotlib seaborn statsmodels
```

Install Quarto for report rendering:
→ https://quarto.org/docs/get-started/

### Step 1 — Download Raw Data

Download the following datasets from the [World Bank WDI](https://databank.worldbank.org/source/world-development-indicators) and place them in the corresponding `data/raw/` subfolders:

| Indicator | WDI Code | Subfolder |
|-----------|----------|-----------|
| Adolescent fertility rate | `SP.ADO.TFRT` | `data/raw/Adol_fert/` |
| Under-5 mortality rate | `SH.DYN.MORT` | `data/raw/Mortality/` |
| Life expectancy at birth | `SP.DYN.LE00.IN` | `data/raw/life_expectancy/` |
| GDP per capita (constant 2015 US$) | `NY.GDP.PCAP.KD` | `data/raw/gdp/` |

### Step 2 — Clean the Data

Run the cleaning scripts in order:

```bash
python scripts/clean_u5_mort.py
python scripts/clean_life_expectancy.py
```

Cleaned outputs are saved to `data/processed/`.

### Step 3 — Run Regression and Generate Figures

```bash
python scripts/03_merge_regression.py       # Merge panel + baseline OLS
python scripts/04_figures.py                # Generate raw-scale figures
python scripts/05_log_merge_regression.py   # Log-transformed regression
python scripts/06_log_robustness.py         # Robustness checks
python scripts/07_log_figures.py            # Log-scale figures
python scripts/analysis_robustness.py       # Additional robustness analysis
```

All figures are saved to `outputs/figures/` and tables to `outputs/tables/`.

### Step 4 — Run Analysis Notebooks (Optional)

```bash
jupyter notebook notebooks/Mortality_data.ipynb   # EDA for U5 mortality
jupyter notebook notebooks/Final_project.ipynb    # Full analysis pipeline
```

### Step 5 — Render the Report

```bash
cd report/
quarto render final_report.qmd --to pdf
quarto render final_report.qmd --to html
```

---

## Data Sources

All data are sourced from the **World Bank World Development Indicators (WDI)** database, a publicly available collection of internationally comparable development statistics.

- World Bank (2024). *World Development Indicators*. Washington, D.C.: The World Bank. https://databank.worldbank.org/source/world-development-indicators

---

## Key Findings

- A one-unit increase in under-5 mortality five years prior is associated with **0.275–0.299 additional adolescent births per 1,000 women** (p < 0.001).
- The relationship holds across all four World Bank income groups but is strongest in low-income countries.
- Life expectancy is a significant negative predictor of adolescent fertility (coef ≈ −2.1 to −2.6), independent of lagged mortality and income group.
- Models explain approximately **67–69%** of the variance in adolescent fertility (R² = 0.671–0.686).

---

## Contribution

We followed a structured GitHub workflow with separate branches for data cleaning, analysis, visualisation, and reporting.

- **Yang Lyu**: Led data integration and regression analysis. Built the merged panel dataset, implemented the 5-year lag variable (`u5_mort_lag5`), conducted baseline and full OLS regression, and performed data quality checks including robustness checks and model comparison.

- **Runyan Tian**: Responsible for data cleaning and preprocessing using SQL (DuckDB), including mortality data preparation, country coverage summaries, and decade-level aggregation. Implemented log transformation of skewed variables to improve model stability and OLS assumption compliance.

- **Steven Hu**: Led visualisation and exploratory data analysis. Produced all figures including time series, scatter plots, faceted income-group comparisons, and log-lagged scatter plots. Implemented alternative lag specifications and conducted missingness sensitivity checks.

- **Eric Chen**: Managed data acquisition from the World Bank WDI, wrote the DuckDB SQL cleaning scripts for adolescent fertility and GDP per capita, built the codebook and ER diagram, and coordinated the final Quarto report structure and repository organisation.

All members contributed to the Quarto report writing, reviewed pull requests, and participated in refining the analysis and interpretation. Commit history and lines of code added are visible in the GitHub Insights tab.



---

## Notes

- Raw CSVs are excluded from version control via `.gitignore` as files are too large for GitHub.
- Two countries — Ethiopia (`ETH`) and Venezuela (`VEN`) — had missing `IncomeGroup` values in the WDI metadata and were manually patched based on World Bank classification.
- Countries with more than 30% missing years across 1990–2023 are excluded from the panel.
- Implausible mortality values (`u5_mort > 400`) are flagged as `is_outlier = True` and excluded from analysis but preserved in the cleaned files for transparency.
