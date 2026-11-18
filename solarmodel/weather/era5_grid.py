"""ERA5 on a grid around the site: loading, site sampling and cloud advection.

The ERA5 value Open-Meteo returns for a site is the hourly mean of one 0.25 deg
grid box (~28 km). A cloud moving at 10 m/s crosses that box in under an
hour, so the box mean smears cloud timing by up to about half an hour and
ignores what the surrounding boxes show. This module works on the
neighbourhood instead:

* :func:`sample_site`   nearest, bilinear or Gaussian-weighted value at the site.
* :func:`advect_site`   backward semi-Lagrangian sampling: the field is moved
  with the steering wind between snapshots, and the site value at time tau
  blends the two bracketing snapshots, each sampled where the air at the site
  came from (the advection step used in nowcasting, Germann and Zawadzki 2002,
  as implemented in pysteps). For ERA5's instantaneous fields (cloud cover)
  this gives better hour means at the site; for its hourly means (irradiance)
  it gives a sub-hourly shape that conserves the hour mean. An hourly mean
  already integrates the motion within the hour, so advecting irradiance does
  not improve hourly values.
* steering wind: 500, 700 or 850 hPa, or weighted by the cloud layers
  ("cloud": 850 for low, 700 for medium, 500 for high cloud).

Limit: advection rearranges the clouds ERA5 has; it cannot add clouds ERA5
missed. :mod:`solarmodel.weather.mos` learns the remaining systematic part
from measurements.

Data come from Earthmover's copy of ERA5 or Open-Meteo (download scripts on the
``nwp-study`` branch), as ``<grid dir>/<site>/*.nc``.
ERA5 accumulations (ssrd, fdir, ssrdc) at valid time t cover the hour ending
at t; they are converted to mean W/m2 and keep the hour-ending label, the
same convention as Open-Meteo.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_ACCUM = {"ssrd": "ghi", "fdir": "bhi", "ssrdc": "ghi_clear"}     # J/m2 per hour -> W/m2
_M_PER_DEG = 111_320.0


def _open_any(path: Path):
    import xarray as xr
    raw = Path(path).read_bytes()
    if raw[:2] == b"PK":                       # CDS sometimes zips netCDF output
        out = []
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for n in z.namelist():
                if n.endswith(".nc"):
                    out.append(xr.open_dataset(io.BytesIO(z.read(n)), engine="h5netcdf").load())
        return out
    return [xr.open_dataset(path, engine="h5netcdf").load()]


def _tidy(ds):
    ren = {}
    if "valid_time" in ds.coords:
        ren["valid_time"] = "time"
    if "pressure_level" in ds.coords:
        ren["pressure_level"] = "level"
    ds = ds.rename(ren)
    if "time" in ds.coords and "time" not in ds.indexes and ds["time"].ndim == 1:
        dim = ds["time"].dims[0]
        ds = ds.swap_dims({dim: "time"}) if dim != "time" else ds.set_xindex("time")
    for c in ("number", "expver"):
        if c in ds.coords and c not in ds.dims:
            ds = ds.drop_vars(c)
    if "expver" in ds.dims:                     # ERA5T/ERA5 overlap: keep the first valid
        ds = ds.reduce(np.nanmax, dim="expver")
    return ds


def open_grid(paths) -> "xr.Dataset":
    """Open and merge CDS downloads into one Dataset on (time, latitude, longitude).

    Variables after loading (when present): ghi, bhi, ghi_clear [W/m2]
    (hour-ending means), tcc, lcc, mcc, hcc [0-1], t2m [degC], u10, v10 [m/s],
    u500/v500, u700/v700, u850/v850 [m/s]. Time is UTC, tz-naive.
    """
    import xarray as xr
    parts = []
    for p in sorted(Path(x) for x in ([paths] if isinstance(paths, (str, Path)) else paths)):
        for ds in _open_any(p):
            ds = _tidy(ds)
            if "level" in ds.dims:
                for lev in ds["level"].values:
                    lv = int(round(float(lev)))
                    sub = ds.sel(level=lev).drop_vars("level")
                    parts.append(xr.Dataset({f"{k}{lv}": sub[k] for k in ("u", "v") if k in sub}))
            else:
                parts.append(ds)
    groups: dict = {}
    for d in parts:
        groups.setdefault(tuple(sorted(d.data_vars)), []).append(d)
    cat = []
    for g in groups.values():
        d = xr.concat(g, dim="time") if len(g) > 1 else g[0]
        d = d.sortby("time")
        cat.append(d.isel(time=~d.get_index("time").duplicated()))
    ds = xr.merge(cat, join="outer")
    for k, new in _ACCUM.items():
        if k in ds:
            ds[new] = (ds[k] / 3600.0).clip(min=0)
            ds = ds.drop_vars(k)
    if "t2m" in ds:
        ds["t2m"] = ds["t2m"] - 273.15
    ds = ds.sortby("latitude").sortby("longitude").sortby("time")
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
    if "ghi" in ds and "ghi_clear" in ds:
        ds["kc"] = _kc(ds)
    if "ghi" in ds and "bhi" in ds:
        ds["fb"] = (ds["bhi"] / ds["ghi"].where(ds["ghi"] > 10.0)).clip(0, 1)
    return ds


# Open-Meteo archive (models=era5) variable -> CDS short name; open_grid converts the units back
OPENMETEO_VARS = {"shortwave_radiation": "ssrd", "direct_radiation": "fdir", "cloud_cover": "tcc",
                  "cloud_cover_low": "lcc", "cloud_cover_mid": "mcc", "cloud_cover_high": "hcc",
                  "temperature_2m": "t2m", "wind_speed_10m": None, "wind_direction_10m": None}


def from_openmeteo(responses) -> "xr.Dataset":
    """Open-Meteo archive JSON responses (one per grid point, wind_speed_unit=ms,
    timezone GMT) -> Dataset with the CDS names and units (ssrd/fdir J/m2,
    cloud 0-1, t2m K, u10/v10 m/s), so :func:`open_grid` reads it like a CDS file.
    No clear-sky GHI (:func:`add_ineichen_clearsky` adds it) and no pressure-level winds.
    Open-Meteo radiation is the preceding-hour mean on the hour-ending stamp,
    the same label as ERA5's accumulations (checked against a CDS file to 0.5 W/m2).
    """
    import xarray as xr
    time = pd.DatetimeIndex(responses[0]["hourly"]["time"])
    lats = sorted({round(r["latitude"], 4) for r in responses})
    lons = sorted({round(r["longitude"], 4) for r in responses})
    if len(lats) * len(lons) != len(responses):
        raise ValueError(f"{len(responses)} points do not form a {len(lats)} x {len(lons)} grid")
    shape = (len(time), len(lats), len(lons))
    arr = {k: np.full(shape, np.nan, dtype="float32") for k in ("ssrd", "fdir", "tcc", "lcc", "mcc", "hcc", "t2m", "u10", "v10")}
    for r in responses:
        i, j = lats.index(round(r["latitude"], 4)), lons.index(round(r["longitude"], 4))
        h = {k: np.asarray(v, dtype=float) for k, v in r["hourly"].items() if k != "time"}
        arr["ssrd"][:, i, j] = h["shortwave_radiation"] * 3600.0
        arr["fdir"][:, i, j] = h["direct_radiation"] * 3600.0
        for k in ("cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"):
            arr[OPENMETEO_VARS[k]][:, i, j] = h[k] / 100.0
        arr["t2m"][:, i, j] = h["temperature_2m"] + 273.15
        rad = np.deg2rad(h["wind_direction_10m"])          # direction the wind blows FROM
        arr["u10"][:, i, j] = -h["wind_speed_10m"] * np.sin(rad)
        arr["v10"][:, i, j] = -h["wind_speed_10m"] * np.cos(rad)
    return xr.Dataset({k: (("time", "latitude", "longitude"), v) for k, v in arr.items()},
                      coords=dict(time=time, latitude=lats, longitude=lons),
                      attrs={"source": "ERA5 via Open-Meteo archive API (models=era5)"})


def add_ineichen_clearsky(ds, alt: float, substeps: int = 6) -> "xr.Dataset":
    """Clear-sky GHI [W/m2] from pvlib Ineichen for every grid cell, for grids
    without ERA5's own (Earthmover and Open-Meteo have no ssrdc), and the
    clear-sky index ``kc`` from it. Each value is the mean of ``substeps``
    sub-interval centres in the hour ending at the stamp, the same averaging as
    ERA5's hourly GHI (a single value at the hour centre misses the sunrise and
    sunset hours). ``alt`` is the site altitude, used for all cells.
    """
    import pvlib
    end = pd.DatetimeIndex(ds.time.values).tz_localize("UTC")
    offs = pd.to_timedelta((np.arange(substeps) + 0.5) * 60.0 / substeps - 60.0, unit="min")
    fine = end + offs[0]
    for o in offs[1:]:
        fine = fine.append(end + o)                    # block k holds sub-interval k of every hour
    cs = np.empty((len(end), ds.sizes["latitude"], ds.sizes["longitude"]), dtype="float32")
    for i, la in enumerate(ds.latitude.values):
        for j, lo in enumerate(ds.longitude.values):
            loc = pvlib.location.Location(float(la), float(lo), altitude=alt)
            g = loc.get_clearsky(fine, model="ineichen")["ghi"].to_numpy()
            cs[:, i, j] = g.reshape(substeps, len(end)).mean(axis=0)
    out = ds.assign(ghi_clear=(("time", "latitude", "longitude"), cs))
    out["kc"] = _kc(out)
    return out


def _kc(ds):
    """Clear-sky index; NaN where the clear sky is under 20 W/m2 (low sun)."""
    return (ds["ghi"] / ds["ghi_clear"].where(ds["ghi_clear"] > 20.0)).clip(0, 1.5)


def _fractional_index(coord: np.ndarray, x):
    """Fractional array index of x on a regular ascending coordinate (clipped)."""
    c0, dc = float(coord[0]), float(coord[1] - coord[0])
    return np.clip((np.asarray(x, float) - c0) / dc, 0.0, len(coord) - 1.0)


def _bilinear(field: np.ndarray, fi: np.ndarray, fj: np.ndarray, t_idx: np.ndarray):
    """Bilinear sample of field[t, i, j] at fractional (fi, fj) for each t_idx; NaN-aware."""
    ny, nx = field.shape[1:]
    i0 = np.clip(np.floor(fi).astype(int), 0, ny - 2); j0 = np.clip(np.floor(fj).astype(int), 0, nx - 2)
    di, dj = fi - i0, fj - j0
    w = np.stack([(1 - di) * (1 - dj), (1 - di) * dj, di * (1 - dj), di * dj])
    v = np.stack([field[t_idx, i0, j0], field[t_idx, i0, j0 + 1],
                  field[t_idx, i0 + 1, j0], field[t_idx, i0 + 1, j0 + 1]])
    ok = np.isfinite(v)
    wsum = (w * ok).sum(axis=0)
    return np.where(wsum > 0, np.nansum(w * np.where(ok, v, 0.0), axis=0) / np.where(wsum > 0, wsum, 1), np.nan)


def sample_site(ds, var: str, lat: float, lon: float, method: str = "bilinear",
                sigma_deg: float = 0.25) -> pd.Series:
    """Hourly value of ``var`` at the site: "nearest", "bilinear" or "gaussian"
    (weights exp(-d^2 / 2 sigma^2) over the whole domain)."""
    f = ds[var].transpose("time", "latitude", "longitude").values
    lats, lons = ds.latitude.values, ds.longitude.values
    t = pd.DatetimeIndex(ds.time.values)
    if method == "nearest":
        i, j = int(np.abs(lats - lat).argmin()), int(np.abs(lons - lon).argmin())
        out = f[:, i, j]
    elif method == "bilinear":
        fi = np.full(len(t), _fractional_index(lats, lat)); fj = np.full(len(t), _fractional_index(lons, lon))
        out = _bilinear(f, fi, fj, np.arange(len(t)))
    elif method == "gaussian":
        dy = (lats[:, None] - lat) * _M_PER_DEG
        dx = (lons[None, :] - lon) * _M_PER_DEG * np.cos(np.radians(lat))
        w = np.exp(-(dx ** 2 + dy ** 2) / (2 * (sigma_deg * _M_PER_DEG) ** 2))
        ok = np.isfinite(f)
        out = np.nansum(f * w, axis=(1, 2)) / np.maximum((ok * w).sum(axis=(1, 2)), 1e-12)
        out = np.where(ok.any(axis=(1, 2)), out, np.nan)
    else:
        raise ValueError(f"method={method!r}")
    return pd.Series(out, index=t, name=var)


def steering_wind(ds, lat: float, lon: float, level: str = "700"):
    """(u, v) [m/s] at the site, hourly, for level "500"/"700"/"850" or "cloud"
    (layer-weighted: low cloud 850, medium 700, high 500 hPa)."""
    if level != "cloud":
        return (sample_site(ds, f"u{level}", lat, lon).to_numpy(),
                sample_site(ds, f"v{level}", lat, lon).to_numpy())
    w = {lv: sample_site(ds, c, lat, lon).fillna(0).to_numpy() for lv, c in (("850", "lcc"), ("700", "mcc"), ("500", "hcc"))}
    tot = sum(w.values())
    tot = np.where(tot > 0.05, tot, np.nan)
    u = sum(w[lv] * sample_site(ds, f"u{lv}", lat, lon).to_numpy() for lv in w) / tot
    v = sum(w[lv] * sample_site(ds, f"v{lv}", lat, lon).to_numpy() for lv in w) / tot
    u0, v0 = steering_wind(ds, lat, lon, "700")      # clear sky: fall back to 700 hPa
    return np.where(np.isfinite(u), u, u0), np.where(np.isfinite(v), v, v0)


INSTANT_VARS = {"tcc", "lcc", "mcc", "hcc", "t2m", "u10", "v10"}


def advect_site(ds, var: str, lat: float, lon: float, wind="700", substeps: int = 6,
                kind: str | None = None, max_disp_cells: float | None = None):
    """Backward semi-Lagrangian site series of ``var``.

    The field is frozen and moved with the steering wind U between two
    snapshots. For a time tau between snapshot times c_a and c_b the site value is

        (1 - w) F_a(x - U (tau - c_a)) + w F_b(x + U (c_b - tau)),
        w = (tau - c_a) / 1 h.

    kind : "instant" for fields valid AT the stamp (ERA5 cloud cover, wind,
        temperature): snapshots at c = t, and the hour-ending mean is the
        average over (t - 1 h, t]. This is where advection improves hourly
        values: the site's mean cloud cover over the hour comes from the two
        snapshots moved along the wind instead of the two values at a fixed point.
        "mean" for hour-ending means (ERA5 accumulations: ghi, kc, fb): each
        field stands at c = t - 30 min; the sub-hourly series is then rescaled
        so its hour mean equals the plain (bilinear) hourly value. An hourly
        mean already contains the motion within the hour, so advection only
        adds the sub-hourly shape; re-averaging it would blur adjacent hours.
        Default: "instant" for INSTANT_VARS, else "mean".
    wind : "500" | "700" | "850" | "cloud" | (u, v) arrays [m/s] per hour.
    substeps : sub-intervals per hour (6 -> 10 min).

    Returns (sub-hourly Series at sub-interval centres, hour-ending Series).
    """
    kind = kind or ("instant" if var in INSTANT_VARS else "mean")
    f = ds[var].transpose("time", "latitude", "longitude").values.astype(float)
    lats, lons = ds.latitude.values, ds.longitude.values
    t = pd.DatetimeIndex(ds.time.values)
    n = len(t)
    if isinstance(wind, str):
        u, v = steering_wind(ds, lat, lon, wind)
    else:
        u, v = (np.asarray(x, float) for x in wind)
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    dlat, dlon = float(lats[1] - lats[0]), float(lons[1] - lons[0])
    m_lon = _M_PER_DEG * np.cos(np.radians(lat))
    lim = max_disp_cells if max_disp_cells is not None else (min(len(lats), len(lons)) - 1) / 2.0
    fi0, fj0 = _fractional_index(lats, lat), _fractional_index(lons, lon)

    def sample(k_idx, dt_s, uu, vv):
        # position the air at the site came from (dt_s > 0: the snapshot is older)
        di = np.clip(-vv * dt_s / _M_PER_DEG / dlat, -lim, lim)
        dj = np.clip(-uu * dt_s / m_lon / dlon, -lim, lim)
        return _bilinear(f, fi0 + di, fj0 + dj, k_idx)

    step = 3600.0 / substeps
    offs = (np.arange(substeps) + 0.5) * step - 3600.0     # sub-interval centres in (t-1h, t]
    centre = 0.0 if kind == "instant" else -1800.0         # snapshot time relative to the stamp
    vals = np.full((n, substeps), np.nan)
    k = np.arange(n)
    for s_, off in enumerate(offs):
        rel = off - centre                                  # tau - c_k
        if rel >= 0:
            a, dt_a = k, rel
        else:
            a, dt_a = k - 1, rel + 3600.0
        a_c, b_c = np.clip(a, 0, n - 1), np.clip(a + 1, 0, n - 1)
        w = dt_a / 3600.0
        fa = sample(a_c, np.full(n, dt_a), u[a_c], v[a_c])
        fb = sample(b_c, np.full(n, -(3600.0 - dt_a)), u[b_c], v[b_c])
        both = np.isfinite(fa) & np.isfinite(fb)
        vals[:, s_] = np.where(both, (1 - w) * fa + w * fb, np.where(np.isfinite(fa), fa, fb))
    with np.errstate(all="ignore"):
        hourly = np.nanmean(vals, axis=1)
    if kind == "mean":                                      # conserve the hourly mean
        plain = sample_site(ds, var, lat, lon, "bilinear").to_numpy()
        with np.errstate(all="ignore"):
            ratio = np.where(np.abs(hourly) > 1e-9, plain / hourly, 1.0)
        vals = vals * ratio[:, None]
        hourly = plain
    sub_idx = (t.to_numpy()[:, None] + pd.to_timedelta(offs, unit="s").to_numpy()[None, :]).ravel()
    sub = pd.Series(vals.ravel(), index=pd.DatetimeIndex(sub_idx), name=f"{var}_adv")
    return sub, pd.Series(hourly, index=t, name=f"{var}_adv")


def site_weather(ds, lat: float, lon: float, alt: float, method: str = "bilinear",
                 freq: str = "h", wind: str = "cloud", substeps: int = 6) -> pd.DataFrame:
    """ghi/dhi/dni/temp_air/wind_speed at the site for ``run_chain``.

    method : "nearest" | "bilinear" | "gaussian".
    GHI is the clear-sky index times the grid's clear-sky GHI (ERA5 ssrdc, or
    Ineichen per cell from :func:`add_ineichen_clearsky`).
    freq : "h" hourly (hour-ending labels) or "sub": sub-hourly series from
        mean-conserving advection (``substeps`` per hour, labels at the
        sub-interval centres; run the chain with ``timestamp_label="center"``).
    The direct share fb = BHI/GHI is treated like kc. DHI = GHI (1 - fb),
    DNI = GHI fb / cos(zenith). Index: tz-aware UTC.
    """
    import pvlib
    if freq == "sub":
        kc = advect_site(ds, "kc", lat, lon, wind=wind, substeps=substeps, kind="mean")[0]
        fb = advect_site(ds, "fb", lat, lon, wind=wind, substeps=substeps, kind="mean")[0]
        mid = pd.DatetimeIndex(kc.index)
        hour_label = mid.ceil("h")
        rep = lambda s_: pd.Series(s_.reindex(hour_label).to_numpy(), index=mid)
        cs_era = rep(sample_site(ds, "ghi_clear", lat, lon, "bilinear"))
        eul = rep(sample_site(ds, "ghi", lat, lon, "bilinear"))
        temp = rep(sample_site(ds, "t2m", lat, lon, "bilinear"))
        wspd = rep(np.hypot(sample_site(ds, "u10", lat, lon, "bilinear"), sample_site(ds, "v10", lat, lon, "bilinear")))
        when = mid
    else:
        kc, fb = sample_site(ds, "kc", lat, lon, method), sample_site(ds, "fb", lat, lon, method)
        cs_era = sample_site(ds, "ghi_clear", lat, lon, "bilinear")
        eul = sample_site(ds, "ghi", lat, lon, method)
        temp = sample_site(ds, "t2m", lat, lon, "bilinear")
        wspd = np.hypot(sample_site(ds, "u10", lat, lon, "bilinear"), sample_site(ds, "v10", lat, lon, "bilinear"))
        when = pd.DatetimeIndex(kc.index) - pd.Timedelta(minutes=30)
    t = kc.index
    ghi = (kc * cs_era).where(np.isfinite(kc), eul).clip(lower=0)      # low sun: keep the plain value
    fb = fb.fillna(0.0).clip(0, 1)
    sp = pvlib.solarposition.get_solarposition(pd.DatetimeIndex(when).tz_localize("UTC"), lat, lon, alt)
    cosz = np.cos(np.radians(sp["zenith"].to_numpy()))
    dhi = ghi * (1 - fb)
    dni = pd.Series(np.where(cosz > 0.065, (ghi * fb).to_numpy() / np.maximum(cosz, 0.065), 0.0), index=t)
    out = pd.DataFrame({"ghi": ghi, "dhi": dhi, "dni": dni.clip(lower=0), "temp_air": temp, "wind_speed": wspd})
    out.index = pd.DatetimeIndex(out.index).tz_localize("UTC")
    return out
