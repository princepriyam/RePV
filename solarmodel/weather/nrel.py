"""US irradiance sources from NREL (now NLR): MIDC ground stations and NSRDB PSM v4.

``midc(...)``   measured GHI, DNI, DHI, air temperature and wind at an MIDC station
                (1-min, pvlib's ``read_midc_raw_data_from_nrel``), quality-checked,
                averaged to 10-min step centres or UTC hour-ending labels.
``nsrdb(...)``  NSRDB GOES Aggregated PSM v4 (4 km, 30 min, 1998 onwards; pvlib's
                ``get_nsrdb_psm4_aggregated``) at the site, on the same labels.

Both cache the raw download under ``cache/weather`` (MIDC per month, NSRDB per
year) and never fill gaps: a missing or rejected value is NaN, and an hour is
NaN unless every 10-min step in it is valid. The bench scores every series on
the hours where all of them are valid, so a station gap removes those hours
from the comparison instead of entering it as data.

Timing
------
MIDC 1-min values are taken as averages over the minute ending at the stamp.
NSRDB values before 2018 are instantaneous 4 km pixels at the stamp (centred
30-min averages from 2018). Either way they are point samples of a curve: the
clear-sky index is interpolated to 10-min centres and multiplied back by the
Ineichen clear sky there, so the hour means keep the diurnal shape.

Quality control (BSRN, Long and Shi 2008, "extremely rare" limits), per minute:
  GHI < 1.2 S0 mu^1.2 + 50,  DHI < 0.75 S0 mu^1.2 + 30,  DNI < 0.95 S0 mu^0.2 + 10,
  all > -4 W/m2; and, where DHI is measured (not computed from GHI and DNI),
  closure |GHI - DNI mu - DHI| / GHI < 8 % (Z < 75 deg) or 15 % (75 to 93 deg) for GHI > 50.
Values with the sun below the horizon are set to 0.

Credentials: ``config/secrets.toml`` ``[nrel] api_key, email`` (not committed),
or the environment variables ``NREL_API_KEY`` and ``NREL_EMAIL``.
"""
from __future__ import annotations

import os
import re
import tomllib

import numpy as np
import pandas as pd
import pvlib
import requests

from solarmodel import paths
from solarmodel.weather.utils import cache_file, cached, hour_labels

STEP = "10min"
MIDC_WIND_HEIGHT_M = {"UNLV": 9.144, "BMS": 1.8288}   # "Avg Wind Speed @ 30ft", "@ 6ft"
_NSRDB_PARAMS = ("ghi", "dni", "dhi", "clearsky_ghi", "clearsky_dni", "air_temperature",
                 "wind_speed", "surface_pressure")


def credentials() -> tuple[str, str]:
    """NREL developer API key and email: environment first, then config/secrets.toml."""
    f = paths.CONFIG_DIR / "secrets.toml"
    cfg = tomllib.load(open(f, "rb")).get("nrel", {}) if f.exists() else {}
    key = os.environ.get("NREL_API_KEY") or cfg.get("api_key", "")
    email = os.environ.get("NREL_EMAIL") or cfg.get("email", "")
    if not key or not email:
        raise RuntimeError(f"NSRDB needs an API key and an email: set [nrel] api_key and email in {f} "
                           "(see config/secrets.example.toml) or NREL_API_KEY / NREL_EMAIL")
    return key, email


def _months(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"))


def qc_irradiance(df: pd.DataFrame, lat: float, lon: float, alt: float, dhi_measured: bool) -> pd.DataFrame:
    """BSRN "extremely rare" limits and closure on ``ghi``, ``dni``, ``dhi`` (UTC index, sample
    instants). Rejected values become NaN; night values become 0. Adds ``qc_fail`` (bool)."""
    sp = pvlib.solarposition.get_solarposition(df.index, lat, lon, altitude=alt)
    z = sp["apparent_zenith"].to_numpy()
    mu = np.clip(np.cos(np.radians(z)), 0.0, None)
    s0 = pvlib.irradiance.get_extra_radiation(df.index).to_numpy()
    out = df.copy()
    lim = {"ghi": 1.2 * s0 * mu ** 1.2 + 50, "dhi": 0.75 * s0 * mu ** 1.2 + 30, "dni": 0.95 * s0 * mu ** 0.2 + 10}
    bad = np.zeros(len(df), bool)
    for c, hi in lim.items():
        v = out[c].to_numpy(float)
        bad |= (v < -4) | (v > hi)
    if dhi_measured:
        g, b, d = (out[c].to_numpy(float) for c in ("ghi", "dni", "dhi"))
        with np.errstate(all="ignore"):
            ratio = np.abs(g - b * mu - d) / g
        tol = np.where(z < 75, 0.08, 0.15)
        bad |= (g > 50) & (z < 93) & (ratio > tol)
    night = z >= 90.0
    for c in ("ghi", "dni", "dhi"):
        v = out[c].to_numpy(float).copy()
        v[bad & ~night] = np.nan
        v[night] = 0.0
        out[c] = np.clip(v, 0.0, None)
    out["qc_fail"] = bad & ~night
    return out


def _steps(start: str, end: str) -> pd.DatetimeIndex:
    """STEP centres in the hours of the UTC hour-ending labels start 00:00 .. end 23:00."""
    labels, half = hour_labels(start, end), pd.Timedelta(STEP) / 2
    return pd.date_range(labels[0] - pd.Timedelta("1h") + half, labels[-1] - half, freq=STEP)


def to_hours(steps: pd.DataFrame, labels: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour-ending means of STEP steps; NaN unless every step of the hour is valid."""
    hour = steps.index.ceil("h")
    g = steps.groupby(hour)
    return g.mean().where(g.count() == pd.Timedelta("1h") // pd.Timedelta(STEP)).reindex(labels)


def _whole_hours(steps: pd.DataFrame) -> pd.DataFrame:
    """10-min steps with every hour that has an invalid step set to NaN (bench drops it)."""
    hour = steps.index.ceil("h")
    ok = steps.notna().all(axis=1).groupby(hour).transform("all")
    return steps.where(ok, np.nan)


def _clearsky(times: pd.DatetimeIndex, lat, lon, alt) -> pd.DataFrame:
    return pvlib.location.Location(lat, lon, altitude=alt).get_clearsky(times, model="ineichen")


def _lerp(values: np.ndarray, t_s: np.ndarray, t_c: np.ndarray, max_gap: float) -> np.ndarray:
    """Linear interpolation of samples at ``t_s`` to ``t_c``; NaN where a neighbour is NaN,
    outside the samples, or where the neighbours are more than ``max_gap`` apart."""
    j = np.searchsorted(t_s, t_c, side="right")
    ok = (j > 0) & (j < len(t_s))
    jl, jr = np.clip(j - 1, 0, len(t_s) - 1), np.clip(j, 0, len(t_s) - 1)
    dt = t_s[jr] - t_s[jl]
    w = np.where(dt > 0, (t_c - t_s[jl]) / np.where(dt > 0, dt, 1), 0.0)
    v = values[jl] * (1 - w) + values[jr] * w
    exact = t_s[jl] == t_c
    v = np.where(exact, values[jl], v)
    return np.where((ok | exact) & ((dt <= max_gap) | exact), v, np.nan)


def interpolate_samples(samples: pd.DataFrame, centres: pd.DatetimeIndex, lat, lon, alt,
                        max_gap: str = "35min") -> pd.DataFrame:
    """Point samples of ``ghi``, ``dni``, ``dhi`` to ``centres`` through the clear-sky index
    (GHI and DHI on clear GHI, DNI on clear DNI); other columns linearly. A centre is NaN
    unless both neighbouring samples are valid and at most ``max_gap`` apart."""
    cs_s, cs_c = _clearsky(samples.index, lat, lon, alt), _clearsky(centres, lat, lon, alt)
    t_s = samples.index.as_unit("ns").asi8.astype(float)      # one unit: cached indexes can be us or ns
    t_c = centres.as_unit("ns").asi8.astype(float)
    gap = float(pd.Timedelta(max_gap).value)
    ref = {"ghi": "ghi", "dhi": "ghi", "dni": "dni"}
    out = {}
    for c in samples.columns:
        x = samples[c].to_numpy(float)
        if c in ref:
            den_s, den_c = cs_s[ref[c]].to_numpy(), cs_c[ref[c]].to_numpy()
            with np.errstate(all="ignore"):
                k = np.clip(np.where(den_s > 5, x / den_s, np.where(np.isnan(x), np.nan, 0.0)), 0.0, 2.0)
            kc = _lerp(k, t_s, t_c, gap)
            out[c] = np.where(den_c > 5, kc * den_c, np.where(np.isnan(kc), np.nan, 0.0))
        else:
            out[c] = _lerp(x, t_s, t_c, gap)
    return pd.DataFrame(out, index=centres)


# Fallback column names by instrument, for station files whose headers predate pvlib's variable
# map (e.g. BMS before an instrument change). Sensors that are tilted, spectral or not
# horizontal are never taken as GHI, DNI or DHI.
_MIDC_PATTERNS = {
    "ghi": (r"^Global CMP22", r"^Global CM22", r"^Global PSP", r"^Global CM21", r"^Global\b"),
    "dni": (r"^Direct CHP1", r"^Direct NIP", r"^Direct\b"),
    "dhi": (r"^Diffuse CM22", r"^Diffuse PSP", r"^Diffuse\b"),
    "temp_air": (r"^Tower Dry Bulb Temp", r"Dry Bulb Temp", r"Air Temp"),
    "wind_speed": (r"^Avg Wind Speed @ 6ft", r"^Avg Wind Speed", r"Wind Speed"),
}
_MIDC_NOT = re.compile(r"South|North|East|West|Tilt|\bUV|PAR|IR\b|Spectral|Normal Incidence 2|Ref Cell|Pyrgeometer|Std Dev|Peak|Max|Min", re.I)


def _midc_rename(df: pd.DataFrame, vmap: dict) -> pd.DataFrame:
    df = df.rename(columns=vmap)
    for var, pats in _MIDC_PATTERNS.items():
        if var in df or (var == "dni" and ("dni_chp1" in df or "dni_nip" in df)):
            continue
        for pat in pats:
            hit = [c for c in df.columns if isinstance(c, str) and re.search(pat, c) and not _MIDC_NOT.search(c)
                   and c not in vmap.values()]
            if hit:
                df = df.rename(columns={hit[0]: var})
                print(f"[midc] {var} from column {hit[0]!r}")
                break
    return df


def _midc_month(site_id: str, month: pd.Timestamp) -> pd.DataFrame:
    def build():
        a, b = month, month + pd.offsets.MonthEnd(0)
        vmap = pvlib.iotools.midc.MIDC_VARIABLE_MAP[site_id]
        try:
            parts = [pvlib.iotools.read_midc_raw_data_from_nrel(site_id, a, b, variable_map={}, timeout=120)]
        except requests.HTTPError:                    # a request across an instrument change fails: by day
            parts = []
            for d in pd.date_range(a, b, freq="D"):
                try:
                    parts.append(pvlib.iotools.read_midc_raw_data_from_nrel(site_id, d, d, variable_map={},
                                                                            timeout=120))
                except requests.HTTPError:
                    print(f"[midc] {site_id} {d.date()}: no data")
        raw = pd.concat(parts)
        df = _midc_rename(raw, vmap)
        if "dni" not in df:                           # BMS: two pyrheliometers; CHP1 is primary, NIP backup
            for alt in ("dni_chp1", "dni_nip"):
                if alt in df:
                    df["dni"] = df[alt] if "dni" not in df else df["dni"].fillna(df[alt])
        if "ghi" not in df:                           # never cache a month without irradiance
            raise KeyError(f"MIDC {site_id} {month:%Y-%m}: no global irradiance column; headers: {list(raw.columns)}")
        keep = [c for c in ("ghi", "dni", "dhi", "temp_air", "wind_speed") if c in df]
        df = df[keep].apply(pd.to_numeric, errors="coerce")
        df.index = df.index.tz_convert("UTC")
        return df[~df.index.duplicated()].sort_index()
    path = cache_file("weather", "midc", site_id, month.strftime("%Y-%m"))
    if path.exists() and "ghi" not in pd.read_parquet(path, columns=None).columns:
        print(f"[cache] {path.name} has no irradiance (old header map): downloading again")
        df = build()
        df.to_parquet(path)
    else:
        df = cached(path, build)
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df


def midc_minutes(site_id: str, start: str, end: str) -> pd.DataFrame:
    """Raw 1-min MIDC data (UTC stamps = end of each minute), cached per month."""
    if site_id not in pvlib.iotools.midc.MIDC_VARIABLE_MAP:
        raise KeyError(f"MIDC station {site_id!r} has no pvlib variable map")
    df = pd.concat([_midc_month(site_id, m) for m in _months(start, end)])
    t0 = pd.Timestamp(start, tz="UTC") - pd.Timedelta("1h")
    return df[(df.index > t0) & (df.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta("1D"))]


def midc(site_id: str, start: str, end: str, lat: float, lon: float, alt: float, *,
         hourly: bool = True, min_valid: int = 9) -> pd.DataFrame:
    """QC'd MIDC weather: ``ghi``, ``dni``, ``dhi`` [W/m2], ``temp_air`` [C], ``wind_speed``
    [m/s, scaled to 10 m] on UTC hour-ending labels (``hourly``) or 10-min centres.
    ``lat``/``lon``/``alt`` are the station's, for the QC sun position (the bench passes the site's;
    UNLV is ~10 km away, well inside the limits' tolerance). A 10-min step needs ``min_valid``
    valid minutes."""
    raw = midc_minutes(site_id, start, end)
    dhi_measured = "(calc)" not in " ".join(pvlib.iotools.midc.MIDC_VARIABLE_MAP[site_id])
    raw.index = raw.index - pd.Timedelta("30s")          # minute centres
    q = qc_irradiance(raw, lat, lon, alt, dhi_measured)
    if "wind_speed" in q:
        h = MIDC_WIND_HEIGHT_M.get(site_id, 10.0)
        q["wind_speed"] = q["wind_speed"] * (10.0 / h) ** (1 / 7)     # power law to the chain's 10 m
    cols = [c for c in ("ghi", "dni", "dhi", "temp_air", "wind_speed") if c in q]
    centres = _steps(start, end)
    bins = (q.index - centres[0] + pd.Timedelta(STEP) / 2) // pd.Timedelta(STEP)
    g = q[cols].groupby(bins)
    steps = g.mean().where(g.count() >= min_valid)
    steps.index = centres[0] + steps.index * pd.Timedelta(STEP)
    steps = steps.reindex(centres)
    n_fail = int(q["qc_fail"].sum())
    if n_fail:
        print(f"[midc] {site_id}: {n_fail} minutes rejected by QC")
    if hourly:
        return to_hours(steps, hour_labels(start, end))
    return _whole_hours(steps)


def nsrdb_year(lat: float, lon: float, year: int) -> pd.DataFrame:
    """NSRDB GOES Aggregated PSM v4, 30 min, UTC, one year (cached). pvlib variable names."""
    def build():
        key, email = credentials()
        df, meta = pvlib.iotools.get_nsrdb_psm4_aggregated(
            lat, lon, key, email, year, time_step=30, parameters=_NSRDB_PARAMS,
            leap_day=True, utc=True, map_variables=True, timeout=300)
        df = df.drop(columns=[c for c in ("Year", "Month", "Day", "Hour", "Minute") if c in df])
        df.attrs = {}
        print(f"[nsrdb] cell {meta.get('Latitude')}, {meta.get('Longitude')}, elevation {meta.get('Elevation')} m")
        return df.apply(pd.to_numeric, errors="coerce")
    df = cached(cache_file("weather", "nsrdb_agg_v4", round(lat, 4), round(lon, 4), year), build)
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df


def nsrdb(lat: float, lon: float, alt: float, start: str, end: str, *, hourly: bool = True) -> pd.DataFrame:
    """NSRDB ``ghi``, ``dni``, ``dhi`` [W/m2], ``temp_air`` [C], ``wind_speed`` [m/s] on UTC
    hour-ending labels or 10-min centres, interpolated through the clear-sky index."""
    years = range(pd.Timestamp(start).year - (pd.Timestamp(start).dayofyear == 1),
                  pd.Timestamp(end).year + 1)
    raw = pd.concat([nsrdb_year(lat, lon, y) for y in years if y >= 1998]).sort_index()
    raw = raw[~raw.index.duplicated()]
    cols = [c for c in ("ghi", "dni", "dhi", "temp_air", "wind_speed") if c in raw]
    q = qc_irradiance(raw[cols], lat, lon, alt, dhi_measured=True).drop(columns="qc_fail")
    steps = _whole_hours(interpolate_samples(q, _steps(start, end), lat, lon, alt))
    if hourly:
        return to_hours(steps, hour_labels(start, end))
    return steps
