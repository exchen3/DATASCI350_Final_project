import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

# paths
panel_path = Path("data/processed/wdi_log_merged_panel.csv")
table_dir = Path("outputs/tables")
table_dir.mkdir(parents=True, exist_ok=True)

# load log panel
df = pd.read_csv(panel_path)

# model helper
def run_model(formula, data, needed_cols):
    temp = data.dropna(subset=needed_cols).copy()
    model = smf.ols(formula, data=temp).fit()
    return model

def collect_model_result(model_name, outcome, predictor, model):
    return {
        "model": model_name,
        "outcome": outcome,
        "main_predictor": predictor,
        "coef_main_predictor": model.params.get(predictor),
        "p_value_main_predictor": model.pvalues.get(predictor),
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs)
    }

# original-scale comparison using the same full 1990-2023 merged panel
model_original = run_model(
    "adol_fert ~ u5_mort_lag5 + life_exp + C(income_group)",
    df,
    ["adol_fert", "u5_mort_lag5", "life_exp", "income_group"]
)

# log robustness models
model_log_lag5 = run_model(
    "log_adol_fert ~ log_u5_mort_lag5 + life_exp + C(income_group)",
    df,
    ["log_adol_fert", "log_u5_mort_lag5", "life_exp", "income_group"]
)

model_log_lag3 = run_model(
    "log_adol_fert ~ log_u5_mort_lag3 + life_exp + C(income_group)",
    df,
    ["log_adol_fert", "log_u5_mort_lag3", "life_exp", "income_group"]
)

model_log_lag10 = run_model(
    "log_adol_fert ~ log_u5_mort_lag10 + life_exp + C(income_group)",
    df,
    ["log_adol_fert", "log_u5_mort_lag10", "life_exp", "income_group"]
)

model_log_gdp = run_model(
    "log_adol_fert ~ log_u5_mort_lag5 + life_exp + log_gdp_per_capita + C(income_group)",
    df,
    ["log_adol_fert", "log_u5_mort_lag5", "life_exp", "log_gdp_per_capita", "income_group"]
)

# compact comparison table
comparison = pd.DataFrame([
    collect_model_result(
        "Original scale lag5",
        "adol_fert",
        "u5_mort_lag5",
        model_original
    ),
    collect_model_result(
        "Log-log lag5",
        "log_adol_fert",
        "log_u5_mort_lag5",
        model_log_lag5
    ),
    collect_model_result(
        "Log-log lag3",
        "log_adol_fert",
        "log_u5_mort_lag3",
        model_log_lag3
    ),
    collect_model_result(
        "Log-log lag10",
        "log_adol_fert",
        "log_u5_mort_lag10",
        model_log_lag10
    ),
    collect_model_result(
        "Log-log lag5 + GDP",
        "log_adol_fert",
        "log_u5_mort_lag5",
        model_log_gdp
    )
])

comparison.to_csv(table_dir / "log_model_comparison.csv", index=False)

# full text output
with open(table_dir / "log_robustness_results.txt", "w") as f:
    f.write("LOG-TRANSFORMED ROBUSTNESS ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    f.write("Purpose:\n")
    f.write(
        "This file keeps the original project workflow unchanged and adds "
        "a parallel log-transformed robustness check for skewness in mortality and fertility rates.\n\n"
    )

    f.write("Model 0: Original-scale comparison model\n")
    f.write("Formula: adol_fert ~ u5_mort_lag5 + life_exp + C(income_group)\n")
    f.write(model_original.summary().as_text())
    f.write("\n\n")

    f.write("Model 1: Log-log lag 5 model\n")
    f.write("Formula: log_adol_fert ~ log_u5_mort_lag5 + life_exp + C(income_group)\n")
    f.write(model_log_lag5.summary().as_text())
    f.write("\n\n")

    f.write("Model 2: Log-log lag 3 model\n")
    f.write("Formula: log_adol_fert ~ log_u5_mort_lag3 + life_exp + C(income_group)\n")
    f.write(model_log_lag3.summary().as_text())
    f.write("\n\n")

    f.write("Model 3: Log-log lag 10 model\n")
    f.write("Formula: log_adol_fert ~ log_u5_mort_lag10 + life_exp + C(income_group)\n")
    f.write(model_log_lag10.summary().as_text())
    f.write("\n\n")

    f.write("Model 4: Log-log lag 5 model with GDP control\n")
    f.write("Formula: log_adol_fert ~ log_u5_mort_lag5 + life_exp + log_gdp_per_capita + C(income_group)\n")
    f.write(model_log_gdp.summary().as_text())

print("Log robustness analysis finished.")
print(comparison)
print("Saved:")
print(table_dir / "log_model_comparison.csv")
print(table_dir / "log_robustness_results.txt")
