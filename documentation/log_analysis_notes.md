# Log Analysis Notes

This file documents the parallel log-transformed analysis.

The original workflow remains unchanged. This additional workflow creates log-transformed versions of under-5 mortality and adolescent fertility to check whether the relationship is robust to skewness.

Main log model:

`log_adol_fert ~ log_u5_mort_lag5 + life_exp + C(income_group)`

Additional checks include lag 3, lag 10, and a GDP-control model.

Because all mortality and fertility values are positive in the regression-ready sample, the workflow uses natural log transformation with `np.log()` instead of `np.log1p()`.
