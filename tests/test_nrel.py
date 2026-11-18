"""NREL inputs (MIDC, NSRDB) offline: QC, averaging, interpolation, the bench path."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pvlib
import pytest

from solarmodel import paths
from solarmodel.weather import inputs, nrel

LAT, LON, ALT = 36.1952, -115.1582, 620.0
LOC = pvlib.location.Location(LAT, LON, altitude=ALT)


@pytest.fixture(autouse=True)
def quiet():
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def _clear(times, scale=1.0):
    cs = LOC.get_clearsky(times, model="ineichen")
    return pd.DataFrame({"ghi": cs.ghi * scale, "dni": cs.dni * scale, "dhi": cs.dhi * scale}, index=times)


def test_qc_rejects_impossible_and_zeros_night():
    t = pd.date_range("2011-06-21 00:00", periods=24 * 60, freq="min", tz="UTC")
    df = _clear(t)
    noon = t[np.argmax(df.ghi.to_numpy())]
    df.loc[noon, "ghi"] = 3000.0
    df.loc[t[0], "ghi"] = -50.0                       # night offset
    q = nrel.qc_irradiance(df, LAT, LON, ALT, dhi_measured=False)
    assert np.isnan(q.loc[noon, "ghi"]) and q.loc[noon, "qc_fail"]
    z = pvlib.solarposition.get_solarposition(t, LAT, LON, altitude=ALT).apparent_zenith
    assert (q.ghi[z >= 90] == 0).all()
    assert q.qc_fail.sum() <= 2                       # the clear sky itself passes
    bad = df.copy(); bad["dhi"] = bad.ghi            # closure fails when DHI is measured
    assert nrel.qc_irradiance(bad, LAT, LON, ALT, dhi_measured=True).qc_fail.sum() > 100


def test_nsrdb_hours_match_clear_sky_hour_means(monkeypatch):
    start = end = "2011-06-21"
    samples = _clear(pd.date_range("2011-06-20 23:00", "2011-06-21 23:30", freq="30min", tz="UTC"), 0.8)
    monkeypatch.setattr(nrel, "nsrdb_year", lambda *a: samples)
    hours = nrel.nsrdb(LAT, LON, ALT, start, end)
    labels = inputs.hour_labels(start, end)
    fine = _clear(pd.date_range("2011-06-20 23:00:30", "2011-06-21 23:00", freq="min", tz="UTC") - pd.Timedelta("30s"), 0.8)
    truth = fine.groupby(fine.index.ceil("h")).mean().reindex(labels)
    day = truth.ghi > 100
    assert hours.index.equals(labels) and np.allclose(hours.ghi[day], truth.ghi[day], rtol=0.01)
    s2 = samples.copy(); s2.loc["2011-06-21 18:00", "ghi"] = np.nan    # a missing daytime sample blanks its neighbours' hours only
    monkeypatch.setattr(nrel, "nsrdb_year", lambda *a: s2)
    h2 = nrel.nsrdb(LAT, LON, ALT, start, end)
    assert 1 <= h2.ghi.isna().sum() - hours.ghi.isna().sum() <= 2


def test_midc_minutes_to_hours(monkeypatch):
    t = pd.date_range("2011-06-20 23:00", "2011-06-22 01:00", freq="min", tz="UTC")[1:]
    raw = _clear(t - pd.Timedelta("30s"))
    raw["temp_air"], raw["wind_speed"] = 35.0, 3.0
    raw.loc["2011-06-21 19:10":"2011-06-21 19:15", "ghi"] = np.nan    # 6 missing minutes in one step
    monkeypatch.setattr(nrel, "midc_minutes", lambda *a, **k: raw)
    h = nrel.midc("UNLV", "2011-06-21", "2011-06-21", LAT, LON, ALT)
    assert len(h) == 24 and h.ghi.isna().sum() == 1 and np.isnan(h.loc["2011-06-21 20:00", "ghi"])
    assert h.wind_speed.dropna().iloc[0] == pytest.approx(3.039, abs=1e-3)    # 30 ft anemometer to 10 m, 1/7 law
    s = nrel.midc("UNLV", "2011-06-21", "2011-06-21", LAT, LON, ALT, hourly=False)
    assert len(s) == 24 * 6 and s.loc["2011-06-21 19:05":"2011-06-21 19:55"].isna().all().all()


def test_inputs_need_a_station_and_credentials(monkeypatch, tmp_path):
    class S: lat, lon, alt = LAT, LON, ALT
    with pytest.raises(KeyError):
        inputs.irradiance("midc", "x", S, "2011-06-21", "2011-06-21", {}, station={})
    with pytest.raises(ValueError):
        inputs.irradiance("nsrdb:5min", "x", S, "2011-06-21", "2011-06-21", {})
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("NREL_API_KEY", raising=False); monkeypatch.delenv("NREL_EMAIL", raising=False)
    with pytest.raises(RuntimeError):
        nrel.credentials()
    (tmp_path / "secrets.toml").write_text('[nrel]\napi_key = "k"\nemail = "e@x"\n')
    assert nrel.credentials() == ("k", "e@x")


def test_bench_scores_station_input_on_its_valid_hours(monkeypatch):
    from dataclasses import replace
    from pathlib import Path
    from solarmodel import bench
    data = Path(__file__).parent / "data" / "bench"
    monkeypatch.setattr(paths, "CACHE_DIR", data)
    site = bench.load_site(data / "LasVegas_34.toml")
    site = replace(site, config={**site.config, "station": {"midc": "UNLV"}})
    point = inputs.irradiance("point", site.name, site.system, site.start, site.end, {})
    met = inputs.met("point", site.name, site.system, site.start, site.end, {})
    fake = point.join(met)
    fake.iloc[4000:4500] = np.nan                       # a three-week station outage
    monkeypatch.setattr(nrel, "midc", lambda *a, **k: fake)
    m, h = bench.run([site], ["current"], weather=["point", "midc"])
    assert m.n_hours.nunique() == 1 and m.n_hours.iloc[0] < 3949
    p = m.set_index("input")
    assert p.loc["midc", "e_sim_kwh"] == pytest.approx(p.loc["point", "e_sim_kwh"], rel=1e-9)
