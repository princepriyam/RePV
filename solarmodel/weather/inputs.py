"""Weather inputs the testbench can compare, as spec strings.

Irradiance (GHI, DHI, DNI):
  "point"                      Open-Meteo ERA5 point series (cached)
  "<grid>:nearest|bilinear|gaussian"
                               an ERA5 grid sampled at the site; ``<grid>`` names a
                               folder in the study's ``[grids]`` table
  "<grid>:advected"            10-min series from mean-conserving advection with the
                               cloud-layer wind; the chain runs at 10 min and is
                               averaged back to the hour
  "midc", "midc:10min"         measured irradiance at the site's MIDC station ([station].midc
                               in the site file), QC'd; hour means or 10-min steps
  "nsrdb", "nsrdb:10min"       NSRDB GOES Aggregated PSM v4 at the site (needs
                               config/secrets.toml, see :mod:`solarmodel.weather.nrel`)
Temperature and wind: "point" (Open-Meteo), "<grid>:bilinear" or "midc".

Station and NSRDB inputs keep their gaps as NaN (an hour with any invalid 10-min
step is NaN); the bench scores only the hours where every series is valid.

Grid folders hold ``<site>/*.nc`` (see :func:`era5_grid.open_grid`). Every
series is on UTC hour-ending labels (``labels``), the advected one on 10-min
centres. A missing hour raises.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from solarmodel import paths
from solarmodel.weather import era5_grid, nrel, site as point
from solarmodel.weather.utils import hour_labels, require_complete  # noqa: F401  (hour_labels: public here)

SUBSTEPS = 6                                # advected input: 10-min steps


def _chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Download pieces: the window itself, or calendar years when it spans more than one."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if s.year == e.year:
        return [(start, end)]
    return [(str(max(s, pd.Timestamp(f"{y}-01-01")).date()), str(min(e, pd.Timestamp(f"{y}-12-31")).date()))
            for y in range(s.year, e.year + 1)]


def _point(getter, lat, lon, start, end, labels) -> pd.DataFrame | pd.Series:
    out = pd.concat([getter(lat, lon, a, b) for a, b in _chunks(start, end)])
    out.index = pd.DatetimeIndex(out.index).tz_localize("UTC")
    return require_complete(out.reindex(labels), getter.__name__)


@lru_cache(maxsize=8)
def grid(folder: str, site_name: str, alt: float):
    """The ERA5 grid around a site, with a clear-sky field (ERA5 ssrdc or Ineichen per cell)."""
    files = sorted((paths.PROJECT_ROOT / folder / site_name).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"no grid files in {paths.PROJECT_ROOT / folder / site_name}")
    ds = era5_grid.open_grid(files)
    return ds if "ghi_clear" in ds else era5_grid.add_ineichen_clearsky(ds, alt)


def _grid_of(spec: str, site_name: str, system, grids: dict):
    key, _, method = spec.partition(":")
    if key not in grids:
        raise KeyError(f"input {spec!r}: no grid {key!r} in the study's [grids] {sorted(grids)}")
    return grid(grids[key], site_name, system.alt), method


def _station(station: dict | None, key: str, spec: str) -> str:
    if not station or key not in station:
        raise KeyError(f"input {spec!r}: the site file has no [station] {key} = ...")
    return station[key]


def _nrel(spec: str, system, start: str, end: str, station: dict | None) -> pd.DataFrame:
    kind, _, res = spec.partition(":")
    if res not in ("", "10min"):
        raise ValueError(f"input {spec!r}: use {kind!r} or {kind + ':10min'!r}")
    if kind == "midc":
        return nrel.midc(_station(station, "midc", spec), start, end, system.lat, system.lon, system.alt,
                         hourly=not res)
    return nrel.nsrdb(system.lat, system.lon, system.alt, start, end, hourly=not res)


def irradiance(spec: str, site_name: str, system, start: str, end: str, grids: dict,
               station: dict | None = None) -> pd.DataFrame:
    labels = hour_labels(start, end)
    if spec.partition(":")[0] in ("midc", "nsrdb"):
        return _nrel(spec, system, start, end, station)[["ghi", "dhi", "dni"]]
    if spec == "point":
        irr = _point(point.get_site_irradiance, system.lat, system.lon, start, end, labels)
        return irr.set_axis(["ghi", "dhi", "dni"], axis=1)
    ds, method = _grid_of(spec, site_name, system, grids)
    if method == "advected":
        w = era5_grid.site_weather(ds, system.lat, system.lon, system.alt, freq="sub", substeps=SUBSTEPS)
        keep = (w.index.ceil("h") >= labels[0]) & (w.index.ceil("h") <= labels[-1])
        return require_complete(w.loc[keep, ["ghi", "dhi", "dni"]], spec)
    w = era5_grid.site_weather(ds, system.lat, system.lon, system.alt, method=method)
    return require_complete(w[["ghi", "dhi", "dni"]].reindex(labels), spec)


def met(spec: str, site_name: str, system, start: str, end: str, grids: dict,
        station: dict | None = None) -> pd.DataFrame:
    """Hourly ``temp_air`` [C] and ``wind_speed`` [m/s, 10 m]."""
    labels = hour_labels(start, end)
    if spec == "midc":
        return _nrel(spec, system, start, end, station)[["temp_air", "wind_speed"]]
    if spec == "point":
        return pd.DataFrame({
            "temp_air": _point(point.get_temperature, system.lat, system.lon, start, end, labels),
            "wind_speed": _point(point.get_wind_speed, system.lat, system.lon, start, end, labels)})
    ds, method = _grid_of(spec, site_name, system, grids)
    w = era5_grid.site_weather(ds, system.lat, system.lon, system.alt, method=method)
    return require_complete(w[["temp_air", "wind_speed"]].reindex(labels), spec)


def snow(system, start: str, end: str, snowfall: bool, depth: bool) -> pd.DataFrame:
    """Point snowfall [cm/h] and ERA5-Land snow depth [cm], only those asked for."""
    labels = hour_labels(start, end)
    out = pd.DataFrame(index=labels)
    if snowfall:
        out["snowfall_cm"] = _point(point.get_snowfall, system.lat, system.lon, start, end, labels)
    if depth:
        out["snow_depth_cm"] = _point(point.get_snow_depth, system.lat, system.lon, start, end, labels)
    return out


def join(irr: pd.DataFrame, met: pd.DataFrame, snow: pd.DataFrame) -> pd.DataFrame:
    """Irradiance joined with the hourly temperature, wind and snow inputs. Sub-hourly
    irradiance gets them on its steps: states and instantaneous values are held at the value
    of their hour-ending label; snowfall [cm per hour] is an amount and is spread evenly over
    the steps of its hour, so the hourly total is kept."""
    if irr.index.equals(met.index):
        return irr.join(met).join(snow)
    hour = irr.index.ceil("h")
    step_min = int(round((irr.index[1] - irr.index[0]) / pd.Timedelta(minutes=1)))

    def on_steps(hourly: pd.DataFrame) -> pd.DataFrame:
        out = hourly.reindex(hour).set_axis(irr.index)
        if "snowfall_cm" in out:
            out["snowfall_cm"] = out["snowfall_cm"] * (step_min / 60.0)
        return out
    return irr.join(on_steps(met)).join(on_steps(snow))


def weather(irr_spec: str, met_spec: str, site_name: str, system, start: str, end: str, grids: dict,
            model, station: dict | None = None) -> pd.DataFrame:
    """Chain input for one model: irradiance ``irr_spec``, temperature and wind ``met_spec`` and the
    snow inputs ``model`` needs (see :func:`join`); ``bench.simulate`` runs it, sub-hourly inputs
    included."""
    return join(irradiance(irr_spec, site_name, system, start, end, grids, station),
                met(met_spec, site_name, system, start, end, grids, station),
                snow(system, start, end, snowfall=model.options.snow == "nrel", depth=model.needs_snow_depth))
