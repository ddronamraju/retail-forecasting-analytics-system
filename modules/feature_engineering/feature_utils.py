# modules/feature_engineering/feature_utils.py
import numpy as np
import pandas as pd

def make_features(d: pd.DataFrame, target: str = "Weekly_Sales") -> pd.DataFrame:
    x = d.copy()
    x["weekofyear"] = x.index.isocalendar().week.astype(int)
    x["month"] = x.index.month
    x["year"] = x.index.year

    for lag in [1, 2, 4, 8, 13, 26, 52]:
        x[f"lag_{lag}"] = x[target].shift(lag)

    for win in [4, 8, 12, 24]:
        base = x[target].shift(1)
        x[f"roll_mean_{win}"] = base.rolling(win).mean()
        x[f"roll_std_{win}"] = base.rolling(win).std()
        x[f"roll_min_{win}"] = base.rolling(win).min()
        x[f"roll_max_{win}"] = base.rolling(win).max()

    x["diff_1"] = x[target].diff(1)
    x["pct_change_1"] = x[target].pct_change(1)

    if "IsHoliday" in x.columns:
        x["holiday_window_3"] = x["IsHoliday"].rolling(3, center=True).max().fillna(0).astype(int)
    else:
        x["holiday_window_3"] = 0

    x["yoy_change"] = (x[target] - x[target].shift(52)) / x[target].shift(52)

    for k in [1, 2, 3]:
        x[f"fourier_sin_{k}"] = np.sin(2 * np.pi * x["weekofyear"] * k / 52)
        x[f"fourier_cos_{k}"] = np.cos(2 * np.pi * x["weekofyear"] * k / 52)

    return x
