"""Model output statistics (MOS): learn the site's clear-sky index from the
ERA5 neighbourhood.

Predictors per hour (all from ERA5 on the downloaded grid), in three nested
sets so their value can be tested (see :func:`neighbourhood_features`):
  * site: clear-sky index, direct share and cloud cover at the site, solar
    elevation, season;
  * neighbourhood: clear-sky index of every cell in a (2h+1) x (2h+1) window
    at t-1, t, t+1 (cloud fields upwind and downwind, before and after);
  * advection: cloud cover moved with the steering wind to the site over the
    hour, and the 700 hPa wind.
Target: the PV clear-sky index measured at the site, k_pv = P_measured / P_clear,
where P_clear is the chain run with clear-sky irradiance, so module, losses
and metadata errors largely cancel.

Models: "gbm" (sklearn HistGradientBoostingRegressor) or "ridge" (RidgeCV on
standardised predictors). Always scored out of sample: leave one year out.

The analog-ensemble idea (Alessandrini et al. 2015, Applied Energy 157,
95-110) is the nearest-neighbour cousin of this; the GBM learns the same
kind of mapping with less tuning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from solarmodel.weather.era5_grid import sample_site, advect_site, steering_wind


FEATURE_SETS = ("site", "neighbourhood", "neighbourhood_advection")


def neighbourhood_features(ds, lat: float, lon: float, half: int = 2, lags=(-1, 0, 1),
                           wind: str = "cloud", features: str = "neighbourhood_advection") -> pd.DataFrame:
    """Predictor table on the grid's hourly index (tz-naive UTC, hour-ending).

    features:
      "site"           kc, fb at the site; cloud cover (mean of the two
                       bracketing snapshots at the site); solar elevation, season
      "neighbourhood"  + kc of every cell in the (2 half + 1)^2 window at each lag
      "neighbourhood_advection"
                       + hour-mean cloud cover advected with the steering
                         wind (low, medium, high, total) and the 700 hPa wind
    Comparing the three answers whether the neighbourhood and the wind add skill.
    """
    import pvlib
    if features not in FEATURE_SETS:
        raise ValueError(f"features={features!r}; one of {FEATURE_SETS}")
    t = pd.DatetimeIndex(ds.time.values)
    cols = {"kc": sample_site(ds, "kc", lat, lon).to_numpy()}
    if "fb" in ds:
        cols["fb"] = sample_site(ds, "fb", lat, lon).to_numpy()
    for c in ("tcc", "lcc", "mcc", "hcc"):
        if c in ds:
            snap = sample_site(ds, c, lat, lon)
            cols[c] = (0.5 * (snap + snap.shift(1))).to_numpy()
    sp = pvlib.solarposition.get_solarposition(t.tz_localize("UTC") - pd.Timedelta(minutes=30), lat, lon)
    cols["elev"] = sp["apparent_elevation"].to_numpy()
    doy = t.dayofyear.to_numpy()
    cols["doy_s"], cols["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    if features in ("neighbourhood", "neighbourhood_advection"):
        lats, lons = ds.latitude.values, ds.longitude.values
        ic, jc = int(np.abs(lats - lat).argmin()), int(np.abs(lons - lon).argmin())
        i0, i1 = max(ic - half, 0), min(ic + half, len(lats) - 1)
        j0, j1 = max(jc - half, 0), min(jc + half, len(lons) - 1)
        kc = ds["kc"].transpose("time", "latitude", "longitude").values[:, i0:i1 + 1, j0:j1 + 1]
        for di in range(kc.shape[1]):
            for dj in range(kc.shape[2]):
                s_ = pd.Series(kc[:, di, dj], index=t)
                for lag in lags:
                    cols[f"kc_{di - (ic - i0):+d}_{dj - (jc - j0):+d}_l{lag:+d}"] = s_.shift(-lag).to_numpy()
    if features == "neighbourhood_advection":
        for c in ("tcc", "lcc", "mcc", "hcc"):
            if c in ds:
                cols[f"{c}_adv"] = advect_site(ds, c, lat, lon, wind=wind, kind="instant")[1].to_numpy()
        u, w = steering_wind(ds, lat, lon, "700")
        cols["u700"], cols["v700"] = u, w
    return pd.DataFrame(cols, index=t)


def _model(kind: str):
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=True,
                                             validation_fraction=0.15, random_state=0)
    if kind == "ridge":
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-2, 3, 12)))
    raise ValueError(f"kind={kind!r}")


def fit_predict_loyo(X: pd.DataFrame, y: pd.Series, kind: str = "gbm", min_train: int = 500) -> pd.Series:
    """Leave-one-year-out predictions of y (NaN where y is unknown or a year
    cannot be held out). X and y share the index; rows with NaN y are not used
    for training but are predicted."""
    pred = pd.Series(np.nan, index=X.index)
    # Hold out calendar years; a site whose other years cannot train (a single-year
    # window such as Henderson 2022-01-31..2023-01-30) holds out calendar months instead.
    lab = y.notna().to_numpy()
    by_year = all(((X.index.year != yr) & lab).sum() >= min_train for yr in set(y.dropna().index.year))
    fold = (X.index.year if by_year else X.index.year * 100 + X.index.month).to_numpy()
    if not by_year:
        print("  MOS: fewer than two trainable years, leave-one-month-out", flush=True)
    for f in sorted(set(fold[lab])):
        train = (fold != f) & lab
        if train.sum() < min_train:
            continue
        m = _model(kind).fit(X[train], y[train])
        test = fold == f
        pred[test] = m.predict(X[test])
    pred.attrs["fold"] = "year" if by_year else "month"
    return pred.clip(0.0, 1.5)
