"""Daily operational forecasting models."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data.generator import india_holidays

TARGETS = {
    "ed_arrivals": "ED arrivals / day",
    "ed_census": "ED occupied beds (daily mean)",
    "icu_census": "ICU occupied beds (daily mean)",
    "med_census": "Medical ward occupied beds (daily mean)",
    "surg_census": "Surgical ward occupied beds (daily mean)",
}
LAGS = (1, 2, 3, 7, 14, 21, 28, 364)

FEATURE_LABELS = {
    "dow": "Day of week", "month": "Month", "doy_sin": "Season (sin)", "doy_cos": "Season (cos)",
    "is_holiday": "Public holiday", "festival": "Festival (Diwali)", "is_weekend": "Weekend",
    "roll7": "7-day mean", "roll28": "28-day mean", "std7": "7-day volatility",
    "trend7": "Weekly trend",
}
FEATURE_LABELS.update({f"lag{lag}": f"Value {lag}d ago" for lag in LAGS})


def calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    hol = india_holidays(range(dates[0].year, dates[-1].year + 1))
    diwali = {d for d, name in hol.items() if "Diwali" in name}
    return pd.DataFrame({
        "dow": dates.dayofweek,
        "month": dates.month,
        "doy_sin": np.sin(2 * np.pi * dates.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * dates.dayofyear / 365.25),
        "is_holiday": [int(d.date() in hol) for d in dates],
        "festival": [int(d.date() in diwali or (d - pd.Timedelta(days=1)).date() in diwali) for d in dates],
        "is_weekend": (dates.dayofweek >= 5).astype(int),
    }, index=dates)


def make_features(y: pd.Series) -> pd.DataFrame:
    X = calendar(pd.DatetimeIndex(y.index))
    for lag in LAGS:
        X[f"lag{lag}"] = y.shift(lag)
    s1 = y.shift(1)
    X["roll7"] = s1.rolling(7, min_periods=3).mean()
    X["roll28"] = s1.rolling(28, min_periods=7).mean()
    X["std7"] = s1.rolling(7, min_periods=3).std()
    X["trend7"] = X["roll7"] - s1.shift(7).rolling(7, min_periods=3).mean()
    return X


def _future_index(y: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


class SeasonalNaive:
    name = "Seasonal naive"

    def fit(self, y: pd.Series):
        self.y = y
        return self

    def predict(self, horizon: int) -> pd.Series:
        last = self.y.iloc[-7:].to_numpy()
        idx = _future_index(self.y, horizon)
        return pd.Series(np.resize(last, horizon), index=idx)


class Sarima:
    name = "SARIMA"

    def fit(self, y: pd.Series):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.y = y.iloc[-400:].asfreq("D")
        ex = calendar(self.y.index)[["is_holiday", "festival"]].astype(float)
        # Fallback chain: numerical issues occasionally break the stationary initialisation.
        specs = [dict(order=(2, 0, 0), seasonal_order=(1, 0, 1, 7)),
                 dict(order=(2, 0, 0), seasonal_order=(1, 0, 1, 7), enforce_stationarity=False),
                 dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, 7))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi = self.y.min() * 0.5, self.y.max() * 1.5
            for spec in specs:
                try:
                    self.res = SARIMAX(self.y, exog=ex, trend="c", **spec).fit(disp=False, maxiter=80)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                if self._sane(lo, hi):
                    return self
            # every spec failed or diverged: fall back to a plain weekly seasonal model
            self.res = SARIMAX(self.y, order=(0, 0, 0), seasonal_order=(0, 1, 1, 7)).fit(disp=False)
            self.exog_free = True
        return self

    exog_free = False

    def _sane(self, lo: float, hi: float) -> bool:
        f = self.predict(28)
        return bool(np.isfinite(f).all() and (f > lo).all() and (f < hi).all())

    def predict(self, horizon: int) -> pd.Series:
        idx = _future_index(self.y, horizon)
        ex = None if self.exog_free else calendar(idx)[["is_holiday", "festival"]].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = self.res.get_forecast(horizon, exog=ex).predicted_mean
        return pd.Series(f.to_numpy(), index=idx)


class LightGBMRecursive:
    name = "LightGBM"
    params = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=12,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  verbose=-1, random_state=0)

    def fit(self, y: pd.Series):
        self.y = y
        X = make_features(y)
        ok = X["lag28"].notna()
        self.features = list(X.columns)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X[ok], y[ok])
        return self

    def predict(self, horizon: int, return_features: bool = False):
        idx = _future_index(self.y, horizon)
        ext = pd.concat([self.y, pd.Series(np.nan, index=idx)])
        rows = []
        for d in idx:
            X = make_features(ext.loc[: d]).iloc[[-1]][self.features]
            ext.loc[d] = float(self.model.predict(X)[0])
            rows.append(X)
        pred = ext.loc[idx]
        if return_features:
            return pred, pd.concat(rows)
        return pred


MODELS = {"Seasonal naive": SeasonalNaive, "SARIMA": Sarima, "LightGBM": LightGBMRecursive}
