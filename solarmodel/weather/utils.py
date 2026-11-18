from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from solarmodel import paths

_OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def hour_labels(start: str, end: str) -> pd.DatetimeIndex:
    """UTC hour-ending labels of the dates ``start``..``end``, as Open-Meteo labels them."""
    return pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h", tz="UTC")


def cache_file(folder: str, *parts) -> Path:
    """``cache/<folder>/<part>__<part>....parquet``."""
    return paths.cache_dir(folder) / ("__".join(str(p) for p in parts) + ".parquet")


def cached(path: Path, build) -> pd.DataFrame:
    """The frame cached at ``path``, or ``build()`` it and cache it."""
    if path.exists():
        print(f"[cache] hit → {path.name}")
        return pd.read_parquet(path)
    df = build()
    df.to_parquet(path)
    print(f"[cache] saved → {path.name}")
    return df


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def require_complete(x: pd.DataFrame | pd.Series, what: str):
    """Refuse gaps instead of filling them: a filled gap would be cached or scored as data.
    A download gap is usually dates not yet published, or an API gap."""
    gaps = x.isna()
    n = int(gaps.any(axis=1).sum() if gaps.ndim == 2 else gaps.sum())
    if n:
        cols = f" in {x.columns[gaps.any()].tolist()}" if gaps.ndim == 2 else ""
        raise RuntimeError(f"{what}: {n} missing hours{cols}")
    return x


def _get(url: str, params: dict) -> dict:
    resp = _session().get(url, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _cached(path: Path, time_col: str, build) -> pd.DataFrame:
    frame = cached(path, build)
    frame[time_col] = pd.to_datetime(frame[time_col])      # old caches may hold the stamps as text
    return frame


def fetch_irradiance(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Hourly GHI, DHI, DNI [W/m2] from Open-Meteo ERA5: columns time, ghi_w_m2, dhi_w_m2, dni_w_m2
    and the requested and served coordinates.

    models=era5: the archive's default blend is ERA5 through 2016 and a different
    source from ~2017 (a 2017/18 step at 11 of 12 study sites).
    """
    def build():
        payload = _get(_OPENMETEO_URL, {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
            "timezone": "UTC",
            "models": "era5",
        })
        hourly = payload["hourly"]
        frame = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"]),
            "ghi_w_m2": pd.array(hourly["shortwave_radiation"], dtype=float),
            "dhi_w_m2": pd.array(hourly["diffuse_radiation"], dtype=float),
            "dni_w_m2": pd.array(hourly["direct_normal_irradiance"], dtype=float),
        })
        require_complete(frame, f"Open-Meteo irradiance {lat}, {lon}")
        frame["requested_lat"] = lat
        frame["requested_lon"] = lon
        frame["source_lat"] = float(payload["latitude"])
        frame["source_lon"] = float(payload["longitude"])
        frame["source"] = "OpenMeteo-era5"
        return frame[["time", "ghi_w_m2", "dhi_w_m2", "dni_w_m2",
                      "requested_lat", "requested_lon", "source_lat", "source_lon", "source"]]
    raw = _cached(cache_file("weather", "openmeteo_irr_era5", lat, lon, start_date, end_date), "time", build)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
    raw = raw[(raw["time"] >= pd.Timestamp(start_date)) & (raw["time"] <= end_ts)].copy()
    if raw.empty:
        raise RuntimeError(f"No data returned for ({lat}, {lon}) in [{start_date}, {end_date}]")
    return raw


def fetch_met(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly 2 m air temperature [°C] and 10 m wind speed [m/s]
    from Open-Meteo for a single location.

    Returns a DataFrame with columns: time, temperature_2m, wind_speed_10m.
    """
    def build():
        hourly = _get(_OPENMETEO_URL, {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,wind_speed_10m",
            "timezone": "UTC",
            # Open-Meteo defaults to km/h; downstream names the column m/s.
            "wind_speed_unit": "ms",
        })["hourly"]
        frame = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"]),
            "temperature_2m": pd.array(hourly["temperature_2m"], dtype=float),
            "wind_speed_10m": pd.array(hourly["wind_speed_10m"], dtype=float),
        })
        return require_complete(frame, f"Open-Meteo temperature/wind {lat}, {lon}")
    return _cached(cache_file("weather", "openmeteo_met", lat, lon, start_date, end_date), "time", build)


def fetch_snow_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Hourly snowfall [cm/h] from Open-Meteo ERA5. Returns columns: time, snowfall_cm.

    Ground snow depth comes from :func:`fetch_snow_depth_era5land`: Open-Meteo
    serves no snow depth for ERA5 itself.
    """
    def build():
        hourly = _get(_OPENMETEO_URL, {
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": end_date,
            "hourly": "snowfall", "models": "era5", "timezone": "UTC",
        })["hourly"]
        frame = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"]),
            "snowfall_cm": pd.array(hourly["snowfall"], dtype=float),
        })
        return require_complete(frame, f"ERA5 snowfall {lat}, {lon}")
    # older cache files also hold snow_depth/temperature columns; only snowfall is used
    return _cached(cache_file("weather", "era5_snow", lat, lon, start_date, end_date), "time", build)[["time", "snowfall_cm"]]


def fetch_snow_depth_era5land(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Hourly ground snow depth [cm] from ERA5-Land (Open-Meteo, models=era5_land).

    Instantaneous values at the stamp. Open-Meteo notes that ERA5-Land snow
    depth tends to be overestimated; check it against a station if snow
    losses matter. Returns columns: time, snow_depth_cm.
    """
    def build():
        hourly = _get(_OPENMETEO_URL, {
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": end_date,
            "hourly": "snow_depth", "models": "era5_land", "timezone": "UTC",
        })["hourly"]
        frame = pd.DataFrame({
            "time": pd.to_datetime(hourly["time"]),
            "snow_depth_cm": pd.array(hourly["snow_depth"], dtype=float) * 100.0,
        })
        require_complete(frame, f"ERA5-Land snow depth {lat}, {lon}")
        frame["snow_depth_cm"] = frame["snow_depth_cm"].clip(lower=0.0)
        return frame
    return _cached(cache_file("weather", "era5land_snowdepth", lat, lon, start_date, end_date), "time", build)


def fetch_soiling_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily precipitation sum [mm/day] for the Kimber soiling model (ERA5 only).

    Returns a DataFrame with columns: date, rain_mm_day.

    models=era5 keeps the soiling input on one reanalysis for all 25 years; the
    archive default blends models and switches source around 2017/18. The cache
    key carries the model so series fetched from the blend are not reused.
    """
    def build():
        daily = _get(_OPENMETEO_URL, {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum",
            "timezone": "UTC",
            "models": "era5",
        })["daily"]
        return require_complete(pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "rain_mm_day": pd.array(daily["precipitation_sum"], dtype=float),
        }), f"ERA5 precipitation {lat}, {lon}")
    return _cached(cache_file("weather", "openmeteo_precip_era5", lat, lon, start_date, end_date), "date", build)
