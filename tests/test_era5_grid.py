"""Tests for ERA5 grid loading, site sampling, advection and MOS (synthetic data)."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("h5netcdf")

from solarmodel.weather import era5_grid as eg
from solarmodel.weather import mos

LAT, LON = 36.03, -114.92
U, V = 15.0, 0.0                               # cloud motion [m/s], eastward
M = 6_371_000.0 * np.pi / 180.0                # metres per degree on a sphere of the Earth's mean radius


def _pattern(x_m, y_m):
    """Smooth periodic cloud field in [0.2, 1.0] (clear-sky index)."""
    return 0.6 + 0.4 * np.sin(2 * np.pi * x_m / 60_000.0) * np.cos(2 * np.pi * y_m / 90_000.0)


def synthetic(hours=24 * 60, n=13, start="2020-01-01 01:00", truth_shift_m=0.0):
    """ERA5-like grid: kc is the box- and hour-mean of a frozen field moving
    with (U, V) (like ERA5 accumulations); tcc is the instantaneous point value
    at the stamp (like ERA5 cloud cover). Returns the grid and the true
    hour-mean kc at the site, optionally taken ``truth_shift_m`` east of it
    (an ERA5 cloud field displaced by that distance)."""
    lats = np.round(np.arange(LAT - 1.5, LAT + 1.5001, 0.25)[:n], 4)
    lons = np.round(np.arange(LON - 1.5, LON + 1.5001, 0.25)[:n], 4)
    t = pd.date_range(start, periods=hours, freq="h")
    t_s = (t - t[0]).total_seconds().to_numpy()
    sub = (np.arange(12) + 0.5) / 12 * 3600 - 3600            # 5-min samples in (t-1h, t]
    box = (np.arange(4) + 0.5) / 4 * 0.25 - 0.125            # 4x4 samples per cell
    my = lambda la: (la - LAT) * M
    mx = lambda lo: (lo - LON) * M * np.cos(np.radians(LAT))
    kc = np.zeros((hours, n, n))
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            ys = my(la + box)[None, :, None, None]
            xs = mx(lo + box)[None, None, :, None]
            tau = (t_s[:, None] + sub[None, :])[:, None, None, :]
            kc[:, i, j] = _pattern(xs - U * tau, ys - V * tau).mean(axis=(1, 2, 3))
    tau = t_s[:, None] + sub[None, :]
    truth = pd.Series(_pattern(truth_shift_m - U * tau, 0.0 - V * tau).mean(axis=1), index=t)
    inst = np.stack([_pattern(mx(lons)[None, :] - U * ts, my(lats)[:, None] - V * ts) for ts in t_s])
    cs = 800.0 * np.clip(np.sin(np.pi * ((t.hour.to_numpy() - 0.5) - 6) / 12), 0, None)
    ghi_clear = np.broadcast_to(cs[:, None, None], kc.shape).copy()
    dims = ("time", "latitude", "longitude")
    ds = xr.Dataset({
        "ghi": (dims, kc * ghi_clear), "ghi_clear": (dims, ghi_clear), "bhi": (dims, 0.5 * kc * ghi_clear),
        "kc": (dims, np.where(ghi_clear > 20, kc, np.nan)), "fb": (dims, np.full_like(kc, 0.5)),
        "tcc": (dims, 1 - inst), "lcc": (dims, np.zeros_like(kc)), "mcc": (dims, 1 - inst), "hcc": (dims, np.zeros_like(kc)),
        "t2m": (dims, np.full_like(kc, 20.0)), "u10": (dims, np.full_like(kc, 3.0)), "v10": (dims, np.zeros_like(kc)),
        **{f"{c}{lv}": (dims, np.full_like(kc, U if c == "u" else V)) for c in "uv" for lv in (500, 700, 850)},
    }, coords={"time": t, "latitude": lats, "longitude": lons})
    return ds, truth


def test_bilinear_exact_on_linear_field():
    ds, _ = synthetic(hours=5)
    ds["lin"] = (("time", "latitude", "longitude"),
                 np.broadcast_to(3.0 * ds.latitude.values[None, :, None] + 2.0 * ds.longitude.values[None, None, :],
                                 (5, ds.sizes["latitude"], ds.sizes["longitude"])).copy())
    v = eg.sample_site(ds, "lin", LAT + 0.07, LON - 0.11, "bilinear")
    assert np.allclose(v, 3.0 * (LAT + 0.07) + 2.0 * (LON - 0.11))


def test_advection_of_instant_field_beats_fixed_point():
    """Hour-mean cloud cover at the site from two hourly snapshots: moving them
    with the wind beats averaging the two values at a fixed point."""
    ds, truth = synthetic()
    true_tcc = 1 - truth
    snap = eg.sample_site(ds, "tcc", LAT, LON, "bilinear")
    fixed = 0.5 * (snap + snap.shift(1))
    _, adv = eg.advect_site(ds, "tcc", LAT, LON, wind="700", substeps=12, kind="instant")
    ok = fixed.notna()
    e_fixed = (fixed - true_tcc)[ok].abs().mean()
    e_adv = (adv - true_tcc)[ok].abs().mean()
    assert e_adv < 0.5 * e_fixed, (e_fixed, e_adv)


def test_mean_field_advection_conserves_hour_mean():
    ds, _ = synthetic(hours=72)
    sub, hourly = eg.advect_site(ds, "kc", LAT, LON, wind="700", substeps=6)
    bil = eg.sample_site(ds, "kc", LAT, LON, "bilinear")
    assert np.allclose(hourly, bil, equal_nan=True)
    per_hour = sub.groupby(np.repeat(np.arange(len(bil)), 6)).mean().to_numpy()
    ok = np.isfinite(per_hour) & np.isfinite(bil.to_numpy())
    assert np.allclose(per_hour[ok], bil.to_numpy()[ok])


def test_advection_without_wind_equals_time_interpolation():
    ds, _ = synthetic(hours=48)
    zero = (np.zeros(ds.sizes["time"]), np.zeros(ds.sizes["time"]))
    sub, hourly = eg.advect_site(ds, "tcc", LAT, LON, wind=zero, substeps=2, kind="instant")
    snap = eg.sample_site(ds, "tcc", LAT, LON, "bilinear")
    # no motion: samples at t-45 min and t-15 min of a linear blend of t-1 h and t
    expect = 0.5 * (0.25 * snap.shift(1) + 0.75 * snap) + 0.5 * (0.75 * snap.shift(1) + 0.25 * snap)
    ok = expect.notna()
    assert np.allclose(hourly[ok], expect[ok], atol=1e-9)


def test_cloud_weighted_wind_falls_back_to_700():
    ds, _ = synthetic(hours=10)
    ds["mcc"] = ds["mcc"] * 0.0
    u, v = eg.steering_wind(ds, LAT, LON, "cloud")
    assert np.allclose(u, U) and np.allclose(v, V)


def _write_cds_like(tmp_path, ds):
    """Write files the way the CDS delivers them (valid_time, pressure_level, J/m2, K)."""
    acc = xr.Dataset({"ssrd": ds["ghi"] * 3600.0, "fdir": ds["bhi"] * 3600.0, "ssrdc": ds["ghi_clear"] * 3600.0})
    inst = xr.Dataset({k: ds[k] for k in ("tcc", "lcc", "mcc", "hcc", "u10", "v10")})
    inst["t2m"] = ds["t2m"] + 273.15
    pl = xr.Dataset({c: xr.concat([ds[f"{c}{lv}"] for lv in (500, 700, 850)], dim="pressure_level")
                     .assign_coords(pressure_level=[500.0, 700.0, 850.0]) for c in "uv"})
    paths = []
    for name, d in (("accum", acc), ("inst", inst), ("pl", pl)):
        d = d.rename(time="valid_time").assign_coords(number=0)
        d = d.sortby("latitude", ascending=False)          # CDS order: north to south
        p = tmp_path / f"2020_{name}.nc"
        d.to_netcdf(p, engine="h5netcdf")
        paths.append(p)
    # one of them zipped, as the CDS sometimes does
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.write(paths[1], arcname="data_stream-oper_stepType-instant.nc")
    paths[1].write_bytes(buf.getvalue())
    return paths


def test_open_grid_roundtrip(tmp_path):
    ds, _ = synthetic(hours=30)
    g = eg.open_grid(_write_cds_like(tmp_path, ds))
    for v in ("ghi", "bhi", "ghi_clear", "kc", "fb", "tcc", "t2m", "u700", "v850"):
        assert v in g, v
    assert np.allclose(g["ghi"].values, ds["ghi"].values)
    assert np.allclose(g["t2m"].values, 20.0)
    assert np.all(np.diff(g.latitude.values) > 0)


def test_site_weather_frame():
    ds, _ = synthetic(hours=48)
    w = eg.site_weather(ds, LAT, LON, 700.0, method="bilinear")
    assert set(w.columns) == {"ghi", "dhi", "dni", "temp_air", "wind_speed"}
    assert str(w.index.tz) == "UTC" and (w["ghi"] >= 0).all()
    assert np.allclose((w["dhi"] / w["ghi"])[w["ghi"] > 50], 0.5)
    sub = eg.site_weather(ds, LAT, LON, 700.0, freq="sub", substeps=6)
    assert len(sub) == 6 * len(w)
    hour = pd.DatetimeIndex(sub.index).ceil("h")
    m = sub["ghi"].groupby(hour).mean().reindex(w.index)
    ok = (w["ghi"] > 50).to_numpy()
    assert np.allclose(m.to_numpy()[ok], w["ghi"].to_numpy()[ok], rtol=1e-6)


def test_mos_neighbourhood_learns_displacement():
    """ERA5 clouds one cell (28 km) out of place: site-only MOS cannot fix it,
    the neighbourhood set can."""
    ds, truth = synthetic(hours=24 * 365 * 2 + 24 * 20, start="2019-01-01 01:00",
                          truth_shift_m=0.25 * M * np.cos(np.radians(LAT)))
    day = ds["ghi_clear"].isel(latitude=0, longitude=0).to_series() > 20
    y = truth.where(day)
    err = {}
    for f in ("site", "neighbourhood"):
        X = mos.neighbourhood_features(ds, LAT, LON, half=2, features=f)
        pred = mos.fit_predict_loyo(X, y, kind="ridge")
        ok = pred.notna() & day
        err[f] = (pred - truth)[ok].abs().mean()
    near = eg.sample_site(ds, "kc", LAT, LON, "nearest")
    e_near = (near - truth)[ok].abs().mean()
    assert err["neighbourhood"] < 0.5 * err["site"] and err["neighbourhood"] < 0.5 * e_near, (err, e_near)


def test_openmeteo_grid_round_trip(tmp_path):
    """Open-Meteo JSON -> CDS-named file -> open_grid gives back W/m2, degC and
    u/v with the meteorological sign (wind FROM the north blows towards -v)."""
    t = [f"2020-06-01T{h:02d}:00" for h in range(24)]
    resp = []
    for la in (36.0, 36.25):
        for lo in (-115.0, -114.75):
            h = {k: [0.0] * 24 for k in eg.OPENMETEO_VARS}
            h.update(shortwave_radiation=[500.0] * 24, direct_radiation=[400.0] * 24, cloud_cover=[50.0] * 24,
                     temperature_2m=[30.0] * 24, wind_speed_10m=[5.0] * 24, wind_direction_10m=[0.0] * 24)
            resp.append(dict(latitude=la, longitude=lo, hourly=dict(time=t, **h)))
    eg.from_openmeteo(resp).to_netcdf(tmp_path / "2020_openmeteo.nc", engine="h5netcdf")
    ds = eg.open_grid([tmp_path / "2020_openmeteo.nc"])
    assert ds.sizes == {"time": 24, "latitude": 2, "longitude": 2}
    assert np.allclose(ds.ghi, 500.0) and np.allclose(ds.bhi, 400.0) and np.allclose(ds.fb, 0.8)
    assert np.allclose(ds.tcc, 0.5) and np.allclose(ds.t2m, 30.0, atol=1e-4)
    assert np.allclose(ds.u10, 0.0, atol=1e-6) and np.allclose(ds.v10, -5.0)


def test_ineichen_clearsky_fallback():
    """A grid without ERA5 clear sky gets pvlib Ineichen per cell as the hour
    mean: GHI at half the 1-min-averaged Ineichen gives kc = 0.5 wherever the
    clear sky is above 50 W/m2, including the sunrise and sunset hours."""
    import pvlib
    t = pd.date_range("2020-06-01 01:00", periods=48, freq="h")
    lat, lon = np.array([36.0, 36.25]), np.array([-115.0, -114.75])
    fine = pd.date_range(t[0] - pd.Timedelta(minutes=59), t[-1], freq="min", tz="UTC")

    def hour_mean(la, lo):
        g = pvlib.location.Location(la, lo, altitude=600).get_clearsky(fine, model="ineichen")["ghi"]
        return g.resample("h", closed="right", label="right").mean().reindex(t.tz_localize("UTC")).to_numpy()

    cs = np.stack([np.stack([hour_mean(la, lo) for lo in lon], -1) for la in lat], -2)
    ds = xr.Dataset({"ghi": (("time", "latitude", "longitude"), 0.5 * cs)},
                    coords=dict(time=t, latitude=lat, longitude=lon))
    out = eg.add_ineichen_clearsky(ds, alt=600)
    up = out["ghi_clear"].values > 50
    assert up.sum() > 50 and np.allclose(out["kc"].values[up], 0.5, atol=0.01)
    assert np.isnan(out["kc"].values[out["ghi_clear"].values <= 20]).all()
