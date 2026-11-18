"""Testbench and model definitions, offline on one site-year (Las Vegas 34, 2011).

tests/data/bench holds the thesis-era weather cache for that year (Open-Meteo
ERA5 irradiance, temperature, wind in the thesis's units, snowfall, rain),
ERA5-Land snow depth, the PVGIS horizon, the measured AC power and the stored
thesis validation series (studies/pv_validation/Validation Freeze, column C).
"""
from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solarmodel import bench, models, paths
from solarmodel.weather import inputs

DATA = Path(__file__).parent / "data" / "bench"
SITE = DATA / "LasVegas_34.toml"


@pytest.fixture(autouse=True)
def fixture_cache(monkeypatch):
    monkeypatch.setattr(paths, "CACHE_DIR", DATA)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_thesis_reproduces_stored_series():
    site = bench.load_site(SITE)
    w = inputs.weather("point", "point", site.name, site.system, site.start, site.end, {}, models.thesis())
    p = bench.simulate(site, models.thesis(), w)["p_ac_kw"]
    stored = pd.read_csv(DATA / "thesis_series.csv", parse_dates=["time"], index_col="time")["p_ac_kw"]
    stored.index = stored.index.tz_localize("UTC")
    np.testing.assert_allclose(p.reindex(stored.index), stored, rtol=0, atol=1e-9)


# recorded 2 Nov 2026 (pvlib 0.15.2, pandas 3.0.5); a change here is a model or scoring change
EXPECTED = {
    "thesis": dict(bias_pct=0.195986491392, nrmse_pct=20.995366069957, kge=0.874618378337),
    "current": dict(bias_pct=-1.285969360011, nrmse_pct=21.026412557521, kge=0.873081959123),
}


def test_scores_are_stable():
    metrics, hourly = bench.run([SITE], ["thesis", "current"])
    assert (metrics.n_hours == 3949).all() and (metrics.lag_h == 0).all() and (metrics.input == "point").all()
    for _, row in metrics.iterrows():
        for k, v in EXPECTED[row.model].items():
            assert row[k] == pytest.approx(v, rel=1e-8), (row.model, k)
    # every model is scored on the same hours, and the yields come from those hours only
    assert hourly[["measured", "thesis|point", "current|point"]].notna().all().all()
    assert metrics.e_meas_kwh.nunique() == 1


def test_model_variants_change_only_what_they_name():
    m = models.parse("current:iam=none:ac_clip=false")
    assert m.name == "current:iam=none:ac_clip=False"
    assert m.options == replace(models.current().options, iam="none") and not m.ac_clip
    assert models.parse("thesis:snow_albedo=null").options == models.thesis().options
    with pytest.raises(TypeError):
        models.current(irradiance="x")               # not a ChainOptions field
    with pytest.raises(ValueError):
        models.parse("current:iam=fresnel")           # not an allowed value
    with pytest.raises(ValueError):
        models.parse("pvgis")                         # unknown base model


def test_best_lag_finds_a_shifted_record():
    idx = pd.date_range("2021-06-01", periods=24 * 20, freq="h", tz="UTC")
    ghi = pd.Series(np.clip(np.sin((idx.hour - 6) / 12 * np.pi), 0, None) * 900, index=idx)
    measured = (ghi * 0.004).shift(-2)                # logged two hours early
    assert bench.best_lag(measured, ghi, max_lag=4) == 2


def test_study_breakdowns_add_up(tmp_path):
    study = tmp_path / "s.toml"
    study.write_text(f"""
sites = ["{SITE}"]
models = ["thesis", "current:iam=none"]
[breakdowns]
month = {{ by = "month" }}
clearness = {{ by = "kc", bins = [0.0, 0.3, 0.6, 0.9, 5.0] }}
""")
    metrics, hourly, out = bench.run_study(study)
    assert out == tmp_path / "outputs" / "s"
    by_month = pd.read_csv(out / "by_month.csv")
    assert set(by_month.month) == set(range(1, 13))
    for _, row in metrics.iterrows():
        m = by_month[by_month.model == row.model]
        assert m.n_hours.sum() == row.n_hours
        assert m.e_sim_kwh.sum() == pytest.approx(row.e_sim_kwh, rel=1e-9)
        assert m.e_meas_kwh.sum() == pytest.approx(row.e_meas_kwh, rel=1e-9)
    assert (out / "by_clearness.csv").exists()


def test_clean_days_drop_outages():
    idx = pd.date_range("2021-06-01 00:30", periods=72, freq="h", tz="UTC")
    ghi_cs = pd.Series(np.clip(np.sin((idx.hour - 13) / 14 * np.pi + np.pi / 2), 0, None) * 1000, index=idx)
    meas = ghi_cs * 0.004
    meas[(idx.day == 2) & (ghi_cs > 300)] = 0.0                     # a day-long outage
    ok = bench.clean_days(meas, ghi_cs, capacity_kw=4.0, tz="UTC")
    assert not ok[idx.day == 2].any() and ok[idx.day != 2].all()
