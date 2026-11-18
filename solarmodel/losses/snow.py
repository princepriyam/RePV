"""Snow on modules: pvlib/NREL coverage model with the inputs reanalysis data needs.

``pvlib.snow.coverage_nrel`` (Marion et al. 2013; Ryberg and Freeman 2017)
fully covers the row when hourly snowfall exceeds ``threshold_snowfall``
(1 cm/h) and lets snow slide off as a function of tilt, irradiance and air
temperature. Two problems appear when it is driven by reanalysis data:

* reanalysis smears snowfall over many hours, so rates above 1 cm/h are rare
  and the model barely triggers. ``trigger="rolling"`` tests the snowfall summed over the
  last ``window_h`` hours instead. This is an engineering adaptation, not a
  published variant of the model: check it against measured snow losses.
* without ground snow depth the model cannot clear the modules when the ground
  is bare. Open-Meteo's ERA5 ``snow_depth`` is empty (null-filled to 0), which
  would clear the modules every hour; use ERA5-Land depth
  (:func:`solarmodel.weather.utils.fetch_snow_depth_era5land`).

``albedo_with_snow`` raises the ground albedo while snow lies on the ground.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def snow_transmittance_nrel(times, snowfall_cm, poa_global, temp_air, surface_tilt, *,
                            snow_depth_cm=None, initial_coverage: float = 0.0,
                            threshold_snowfall: float = 1.0, threshold_depth: float = 1.0,
                            trigger: str = "hourly", window_h: int = 6) -> pd.Series:
    """Hourly snow transmittance ``1 - coverage`` on ``times``.

    With the defaults (hourly trigger, no depth) this is the frozen thesis
    chain's snow model.
    """
    snowfall = snowfall_cm.reindex(times, fill_value=0.0)
    if trigger == "rolling":
        # trailing window of whole hours, as on an hourly index (Eq. snow_trigger):
        # every step of hour H sees the snowfall of hours H-window_h+1 .. H. pvlib
        # divides each value by the step length to get an hourly rate, so the window
        # sum is passed as an amount per step, and the rate equals the window sum.
        hour = times.ceil("h")
        per_hour = snowfall.groupby(hour).sum()
        window = per_hour.rolling(f"{int(window_h)}h", min_periods=1).sum()
        dt_h = float(pd.Series(times).diff().median() / pd.Timedelta(hours=1)) if len(times) > 1 else 1.0
        snowfall = pd.Series(window.reindex(hour).to_numpy() * dt_h, index=times)
    elif trigger != "hourly":
        raise ValueError(f"unknown snow trigger {trigger!r}")
    kw = {}
    if snow_depth_cm is not None:
        kw["snow_depth"] = snow_depth_cm.reindex(times).ffill().fillna(0.0)
        kw["threshold_depth"] = float(threshold_depth)
    if threshold_snowfall != 1.0:
        kw["threshold_snowfall"] = float(threshold_snowfall)
    coverage = pvlib.snow.coverage_nrel(
        snowfall=snowfall,
        poa_irradiance=poa_global.reindex(times).fillna(0.0),
        temp_air=temp_air.reindex(times),
        surface_tilt=surface_tilt if np.ndim(surface_tilt) else float(surface_tilt),
        initial_coverage=float(initial_coverage),
        **kw,
    )
    return 1.0 - coverage.clip(lower=0.0, upper=1.0)


def albedo_with_snow(base_albedo, snow_depth_cm: pd.Series, snow_albedo: float = 0.6,
                     threshold_cm: float = 1.0) -> pd.Series:
    """Ground albedo series: ``snow_albedo`` where ground snow depth >= threshold.

    The 0.6 default is an assumption for aged/patchy snow on farmland; fresh
    snow is higher. Set it per site.
    """
    base = pd.Series(base_albedo, index=snow_depth_cm.index, dtype=float)
    return base.where(snow_depth_cm.fillna(0.0) < threshold_cm, snow_albedo)
