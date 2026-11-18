"""One weather variable per call for a site, indexed by naive UTC time."""
from __future__ import annotations

import pandas as pd
from solarmodel.weather.utils import (fetch_irradiance, fetch_met, fetch_snow_weather,
                                      fetch_snow_depth_era5land, fetch_soiling_weather, require_complete)


def get_site_irradiance(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Hourly GHI, DHI, DNI [W/m2], ERA5: ``ghi_w_m2``, ``dhi_w_m2``, ``dni_w_m2``."""
    frame = fetch_irradiance(lat, lon, start_date, end_date)
    return frame[["time", "ghi_w_m2", "dhi_w_m2", "dni_w_m2"]].set_index("time")


def get_temperature(lat: float, lon: float, start_date: str, end_date: str) -> pd.Series:
    """2 m air temperature [C]."""
    return fetch_met(lat, lon, start_date, end_date).set_index("time")["temperature_2m"]


def get_wind_speed(lat: float, lon: float, start_date: str, end_date: str) -> pd.Series:
    """10 m wind speed [m/s]."""
    return fetch_met(lat, lon, start_date, end_date).set_index("time")["wind_speed_10m"]


def get_snowfall(lat: float, lon: float, start_date: str, end_date: str) -> pd.Series:
    """Hourly snowfall [cm/h], ERA5."""
    return fetch_snow_weather(lat, lon, start_date, end_date).set_index("time")["snowfall_cm"]


def get_snow_depth(lat: float, lon: float, start_date: str, end_date: str) -> pd.Series:
    """Ground snow depth [cm], ERA5-Land, instantaneous at the stamp."""
    return fetch_snow_depth_era5land(lat, lon, start_date, end_date).set_index("time")["snow_depth_cm"]


def hourly_weather(lat: float, lon: float, start_date: str, end_date: str, *,
                   snow_depth: bool, temp_air: pd.Series | None = None,
                   wind_speed: pd.Series | None = None) -> pd.DataFrame:
    """Chain input on a UTC hour-ending index: ERA5 irradiance, temperature and
    wind (Open-Meteo unless given), snowfall, and ERA5-Land snow depth if asked.

    Every input must cover every hour: a gap raises instead of being filled.
    """
    irr = get_site_irradiance(lat, lon, start_date, end_date)
    times = pd.DatetimeIndex(irr.index).tz_localize("UTC")
    near = dict(method="nearest", tolerance=pd.Timedelta("30min"))

    def on_times(s: pd.Series, **kw) -> pd.Series:
        s = s.copy()
        s.index = pd.DatetimeIndex(s.index)
        s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index.tz_convert("UTC")
        return s.reindex(times, **kw)

    w = pd.DataFrame({
        "ghi": irr["ghi_w_m2"].to_numpy(), "dhi": irr["dhi_w_m2"].to_numpy(), "dni": irr["dni_w_m2"].to_numpy(),
        "temp_air": on_times(get_temperature(lat, lon, start_date, end_date) if temp_air is None else temp_air, **near),
        "wind_speed": on_times(get_wind_speed(lat, lon, start_date, end_date) if wind_speed is None else wind_speed, **near),
        "snowfall_cm": on_times(get_snowfall(lat, lon, start_date, end_date)),
    }, index=times)
    if snow_depth:
        w["snow_depth_cm"] = on_times(get_snow_depth(lat, lon, start_date, end_date))
    return require_complete(w, f"weather {lat}, {lon} {start_date}..{end_date}")


def hourly_soiling(spec, lat: float, lon: float, start_date: str, end_date: str,
                   times: pd.DatetimeIndex) -> pd.Series:
    """Daily soiling ratio held over each day, on ``times`` (UTC), from ERA5 daily rain."""
    from solarmodel.losses.soiling_loss import soiling_ratio_daily
    daily = soiling_ratio_daily(spec, fetch_soiling_weather(lat, lon, start_date, end_date))
    return daily.reindex(times, method="ffill").fillna(1.0)   # clean before the first day's value
