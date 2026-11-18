"""PVGIS hourly PV output as a benchmark series (JRC PVGIS 5.3 ``seriescalc``).

PVGIS models the site from its own radiation database and PV model; nothing of
this package's chain is used, so it is the "off-the-shelf tool" reference.
Only PVGIS-ERA5 covers North America (SARAH3 does not; NSRDB is not offered).

Spec in a study: ``benchmarks = ["pvgis"]`` or ``"pvgis:loss=10:mounting=building"``.
Defaults are the PVGIS web defaults: crystalline Si, free-standing, 14 % system loss,
terrain horizon on.
"""
from __future__ import annotations

import pandas as pd

from solarmodel import paths
from solarmodel.weather import cache

DEFAULTS = dict(db="PVGIS-ERA5", loss=14.0, mounting="free", tech="crystSi")


def parse(spec: str) -> dict:
    base, *parts = spec.split(":")
    if base != "pvgis":
        raise ValueError(f"benchmark {spec!r}: only 'pvgis[:key=value...]'")
    o = dict(DEFAULTS)
    for part in parts:
        k, _, v = part.partition("=")
        if k not in o:
            raise ValueError(f"benchmark {spec!r}: {k!r} not one of {sorted(o)}")
        o[k] = float(v) if k == "loss" else v
    return o


def label(spec: str) -> str:
    o = parse(spec)
    extra = "".join(f":{k}={o[k]:g}" if k == "loss" else f":{k}={o[k]}"
                    for k in ("loss", "mounting") if o[k] != DEFAULTS[k])
    return o["db"] + extra


def fetch_year(lat: float, lon: float, year: int, tilt: float, azimuth: float, db: str, key: str, keep,
               **kw) -> pd.DataFrame:
    """One year of PVGIS hourly data on a fixed plane (azimuth in the pvlib convention, terrain
    horizon on), cached as ``cache/pvgis/<db>__<lat>__<lon>__<year>__t<tilt>_a<azimuth>_<key>.parquet``;
    ``keep`` picks what is cached from the pvlib frame, ``kw`` go to ``get_pvgis_hourly``."""
    fp = paths.cache_dir("pvgis") / f"{db}__{lat:.4f}__{lon:.4f}__{year}__t{tilt:g}_a{azimuth:g}_{key}.parquet"
    hit = cache.read(fp)                          # parquet; a legacy CSV is converted and deleted
    if hit is not None:
        return hit
    import pvlib
    d, *_ = pvlib.iotools.get_pvgis_hourly(lat, lon, start=year, end=year, raddatabase=db, surface_tilt=tilt,
                                           surface_azimuth=azimuth, usehorizon=True, **kw)
    return cache.write(fp, keep(d))


def on_labels(d, labels: pd.DatetimeIndex):
    """PVGIS-ERA5 hourly values on hour-ending UTC ``labels``: PVGIS stamps each hour's mean at
    HH:30 UTC (centre of the ERA5 hour), so the label is the stamp + 30 min."""
    idx = pd.DatetimeIndex(d.index)
    d = d.set_axis((idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")) + pd.Timedelta(minutes=30))
    return d[~d.index.duplicated()].reindex(labels)


def power(site_cfg: dict, start: str, end: str, labels: pd.DatetimeIndex, spec: str = "pvgis") -> pd.Series:
    """PVGIS AC power [kW] on the bench's hour-ending UTC ``labels``."""
    o = parse(spec)
    loc, arr, sysd = site_cfg["location"], site_cfg["array"], site_cfg["system"]
    if arr.get("tracking", False):
        raise ValueError("pvgis benchmark: fixed-tilt sites only")
    tilt, az, kwp = float(arr["tilt"]), float(arr["azimuth"]), float(sysd["rated_capacity_kw"])
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    s = pd.concat(fetch_year(loc["lat"], loc["lon"], y, tilt, az, o["db"],
                             f"p{kwp:g}_l{o['loss']:g}_{o['mounting']}_{o['tech']}",
                             lambda d: (d["P"] / 1000.0).rename("p_ac_kw").to_frame(),           # W -> kW
                             components=False, pvcalculation=True, peakpower=kwp, pvtechchoice=o["tech"],
                             mountingplace=o["mounting"], loss=o["loss"]).iloc[:, 0] for y in years)
    return on_labels(s, labels)
