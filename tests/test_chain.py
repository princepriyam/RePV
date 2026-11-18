"""Unit tests for the new chain components (no network, synthetic inputs)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pvlib
import pytest

from solarmodel.model import transposition as tr
from solarmodel.model.chain import ChainOptions, PVSystem, Layout, run_chain, module_report
from solarmodel.weather.horizon import flat_horizon, merge_horizons
from solarmodel.losses.snow import snow_transmittance_nrel, albedo_with_snow

LAT, LON, ALT = 45.78, 16.20, 117.0
IDX = pd.date_range("2021-01-01 01:00", "2021-12-31 23:00", freq="h", tz="UTC")


def _clear_weather(idx=IDX, scale=1.0):
    loc = pvlib.location.Location(LAT, LON, altitude=ALT)
    cs = loc.get_clearsky(idx - pd.Timedelta(minutes=30), model="haurwitz")
    ghi = pd.Series(cs["ghi"].to_numpy(), index=idx) * scale
    sp = loc.get_solarposition(idx - pd.Timedelta(minutes=30))
    d = pvlib.irradiance.erbs(ghi, pd.Series(sp["zenith"].to_numpy(), index=idx), idx)
    return pd.DataFrame({"ghi": ghi, "dhi": d["dhi"], "dni": d["dni"],
                         "temp_air": 15.0, "wind_speed": 3.0, "snowfall_cm": 0.0, "snow_depth_cm": 0.0,
                         "relative_humidity": 70.0}, index=idx)


def _system(**kw):
    m = dict(isc_ref=11.21, voc_ref=49.50, imp_ref=10.53, vmp_ref=42.70, mu_isc=0.00505, mu_voc=-0.140, ns_mod=72)
    base = dict(lat=LAT, lon=LON, alt=ALT, tilt=35.0, azimuth=180.0, albedo=0.2, module=m,
                ns_arr=25, np_arr=40, thermal=dict(u_c=29.0, u_v=0.0, eta_module=0.209, alpha_abs=0.9,
                                                   u0=25.0, u1=6.84))
    base.update(kw)
    return PVSystem(**base)


def _geo(w):
    sp = pvlib.solarposition.get_solarposition(w.index - pd.Timedelta(minutes=30), LAT, LON, ALT)
    sp.index = w.index
    am = pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"])
    return sp, am, pvlib.irradiance.get_extra_radiation(w.index)


def test_muneer_horizontal_equals_dhi():
    w = _clear_weather(); sp, _, de = _geo(w)
    sky = tr.muneer_pvgis(0.0, 180.0, w["dhi"], w["dni"], de, sp["apparent_zenith"], sp["azimuth"])
    np.testing.assert_allclose(sky, w["dhi"], rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("model", ["reindl", "perez", "haydavies", "isotropic"])
def test_flat_horizon_matches_pvlib(model):
    w = _clear_weather(); sp, am, de = _geo(w)
    a = tr.poa_open_field(30.0, 180.0, sp["apparent_zenith"], sp["azimuth"], w["dni"], w["ghi"], w["dhi"],
                          de, am, 0.2, model=model, horizon=None)
    b = tr.poa_open_field(30.0, 180.0, sp["apparent_zenith"], sp["azimuth"], w["dni"], w["ghi"], w["dhi"],
                          de, am, 0.2, model=model, horizon=flat_horizon())
    np.testing.assert_allclose(b["poa_global"], a["poa_global"], rtol=1e-9, atol=1e-6)


def test_horizon_blocks_beam_and_svf():
    w = _clear_weather(); sp, am, de = _geo(w)
    wall = pd.Series(80.0, index=np.arange(0, 360, 7.5))
    r = tr.poa_open_field(30.0, 180.0, sp["apparent_zenith"], sp["azimuth"], w["dni"], w["ghi"], w["dhi"],
                          de, am, 0.2, model="perez", horizon=wall)
    assert float(r["poa_direct"].max()) == 0.0
    h = pd.Series(20.0, index=np.arange(0, 360, 7.5))
    # horizontal plane, uniform 20 deg horizon: remaining isotropic sky = cos^2(20 deg)
    assert tr.horizon_sky_view_ratio(h, 0.0, 180.0) == pytest.approx(np.cos(np.radians(20)) ** 2, rel=2e-3)
    assert tr.horizon_sky_view_ratio(flat_horizon(), 35.0, 180.0) == pytest.approx(1.0)


def test_merge_horizons_envelope():
    a = pd.Series([0.0, 5.0, 0.0, 0.0], index=[0, 90, 180, 270])
    b = pd.Series([2.0, 0.0, 10.0, 0.0], index=[0, 90, 180, 270])
    m = merge_horizons(a, b, step=90.0)
    assert list(m.values) == [2.0, 5.0, 10.0, 0.0]


def test_bad_option_rejected():
    with pytest.raises(ValueError):
        ChainOptions(transposition="nonsense")


def test_presets_run_and_rank():
    w = _clear_weather()
    sys_open = _system()
    sys_rows = _system(layout=Layout(gcr=0.4, height_m=1.5, collector_width_m=2.0))
    e = {}
    e["frozen"] = run_chain(w, sys_open, ChainOptions.frozen())["p_ac_kw"].sum()
    e["diffuse_iam"] = run_chain(w, sys_open, ChainOptions.frozen(iam="physical"))["p_ac_kw"].sum()
    e["rows"] = run_chain(w, sys_rows, ChainOptions.improved())["p_ac_kw"].sum()
    e["bifacial"] = run_chain(w, _system(layout=Layout(), bifaciality=0.7),
                              ChainOptions.improved())["p_ac_kw"].sum()
    e["pvgis"] = run_chain(w, _system(horizon=flat_horizon()), ChainOptions.pvgis_like())["p_ac_kw"].sum()
    assert e["diffuse_iam"] < e["frozen"]           # IAM on diffuse removes light
    assert e["rows"] < e["diffuse_iam"]             # row shading removes more
    assert e["bifacial"] > e["rows"]                # rear side adds some back
    assert 0.85 < e["pvgis"] / e["frozen"] < 1.15   # independent chain, same ballpark


def test_ilr_clipping():
    w = _clear_weather()
    o = ChainOptions.frozen(loss_order="physical")
    lo = run_chain(w, _system(ilr=1.0), o)
    hi = run_chain(w, _system(ilr=1.6), o)
    assert float(hi["clip_kw"].sum()) > 0
    assert float(hi["p_ac_kw"].max()) < float(lo["p_ac_kw"].max())
    assert float(hi["p_ac_kw"].sum()) <= float(lo["p_ac_kw"].sum())


def test_export_limit():
    w = _clear_weather()
    out = run_chain(w, _system(export_limit_kw=200.0), ChainOptions.improved())
    assert float(out["p_ac_kw"].max()) <= 200.0 + 1e-9


def test_module_report_reasonable():
    r = module_report(_system())
    assert -0.45 < r["gamma_pmp_pct_per_k"] < -0.25
    assert 0.85 < r["rel_eff"][200] < 1.02
    assert 440 < r["p_stc_w"] < 460


def test_firstsolar_spectral_needs_rh():
    w = _clear_weather().drop(columns="relative_humidity")
    with pytest.raises(ValueError):
        run_chain(w, _system(), ChainOptions.frozen(spectral="firstsolar"))


def test_snow_rolling_trigger_and_depth():
    idx = pd.date_range("2021-01-10", periods=48, freq="h", tz="UTC")
    snow = pd.Series(0.0, index=idx); snow.iloc[5:11] = 0.4      # 2.4 cm over 6 h, never 1 cm/h
    poa = pd.Series(0.0, index=idx); temp = pd.Series(-3.0, index=idx)
    hourly = snow_transmittance_nrel(idx, snow, poa, temp, 35.0)
    rolling = snow_transmittance_nrel(idx, snow, poa, temp, 35.0, trigger="rolling", window_h=6)
    assert hourly.min() == 1.0 and rolling.min() == 0.0
    bare = snow_transmittance_nrel(idx, snow, poa, temp, 35.0, trigger="rolling",
                                   snow_depth_cm=pd.Series(0.0, index=idx))
    assert bare.iloc[-1] == 1.0                                    # bare ground clears the modules
    alb = albedo_with_snow(0.2, pd.Series([0.0, 5.0], index=idx[:2]))
    assert list(alb) == [0.2, 0.6]


def test_snow_tilt_stand_in_for_horizontal_configs():
    """At tilt 0 NREL snow coverage cannot slide off, so one January snowfall
    would blank the rest of the year: the chain refuses, and a snow_tilt
    stand-in lets the snow clear."""
    w = _clear_weather()
    w.loc["2021-01-10 06:00":"2021-01-10 08:00", "snowfall_cm"] = 5.0
    w["temp_air"] = 2.0
    opt = ChainOptions.frozen(snow="nrel")
    with pytest.raises(ValueError, match="tilt 0"):
        run_chain(w, _system(tilt=0.0), opt)
    fixed = run_chain(w, _system(tilt=0.0, snow_tilt=40.0), opt)["p_ac_kw"]
    assert fixed["2021-06"].sum() > 0.0
    no_snow = run_chain(w, _system(tilt=0.0), replace(opt, snow="none"))["p_ac_kw"]
    assert fixed["2021-01-10"].sum() < no_snow["2021-01-10"].sum()      # covered on the day it fell


def test_improved_needs_snow_depth():
    with pytest.raises(KeyError, match="snow_depth_cm"):
        run_chain(_clear_weather().drop(columns="snow_depth_cm"), _system(), ChainOptions.improved())


def test_sandia_inverter():
    from solarmodel.model.chain import cec_inverter
    inv = cec_inverter("Satcon_Technology__PVS_135__480V_")
    w = _clear_weather()
    opt = ChainOptions.improved(snow="none", snow_albedo=None, use_horizon=False, inverter="sandia")
    sys_ = _system(ns_arr=9, np_arr=30, inverter_params=inv)      # 9 x 42.7 V = 384 V, 121 kWp
    out = run_chain(w, sys_, opt)
    day = out.p_dc_kw > 0.3 * sys_.pdc0 / 1000
    # Satcon PVS-135 curve (94 to 97.5 % from 30 to 100 % load) after about 5 % mismatch and wiring
    assert (out.p_ac_kw[day] / out.p_dc_kw[day]).between(0.89, 0.93).all()
    assert (out.p_ac_kw[out.p_dc_kw == 0] == 0).all()              # no night tare on the meter
    big = run_chain(w, _system(ns_arr=9, np_arr=60, inverter_params=inv), opt)
    assert big.p_ac_kw.max() <= inv["Paco"] / 1000 and big.clip_kw.sum() > 0
    with pytest.raises(ValueError):
        run_chain(w, sys_, replace(opt, loss_order="post_inverter"))
    with pytest.raises(ValueError):
        run_chain(w, _system(ns_arr=9, np_arr=30), opt)            # no [inverter].cec
    with pytest.raises(KeyError):
        cec_inverter("Satcon 135kW")


@pytest.mark.parametrize("trigger", ["hourly", "rolling"])
@pytest.mark.parametrize("with_depth", [False, True])
def test_snow_state_independent_of_timestep(trigger, with_depth):
    """Hourly inputs expanded to 10-min steps as the bench does give the hourly snow
    state at the last step of every hour (events, 6 h window, sliding, depth reset)."""
    from solarmodel.weather.inputs import join
    idx = pd.date_range("2020-01-01 01:00", periods=72, freq="h", tz="UTC")
    snow = pd.Series(0.0, index=idx)
    snow.iloc[[3, 4, 5, 30]] = [0.4, 0.5, 0.3, 1.5]      # a smeared event and a sharp one
    poa = pd.Series(np.where((idx.hour > 14) & (idx.hour < 23), 400.0, 0.0), index=idx)
    temp = pd.Series(np.where(idx.hour % 24 > 12, 2.0, -6.0), index=idx)
    depth = pd.Series(np.where(np.arange(72) < 55, 5.0, 0.0), index=idx)
    kw = dict(trigger=trigger, window_h=6)
    hourly = snow_transmittance_nrel(idx, snow, poa, temp, 35.0,
                                     snow_depth_cm=depth if with_depth else None, **kw)
    steps = pd.date_range(idx[0] - pd.Timedelta("55min"), idx[-1] - pd.Timedelta("5min"), freq="10min")
    x = join(pd.DataFrame(index=steps), pd.DataFrame({"poa": poa, "temp": temp, "depth": depth}),
             pd.DataFrame({"snowfall_cm": snow}))
    assert np.isclose(x["snowfall_cm"].sum(), snow.sum())
    sub = snow_transmittance_nrel(steps, x["snowfall_cm"], x["poa"], x["temp"], 35.0,
                                  snow_depth_cm=x["depth"] if with_depth else None, **kw)
    last = sub[steps.minute == 55]
    assert hourly.lt(1).any() and hourly.eq(0).any()     # the case covers events and sliding
    np.testing.assert_allclose(last.to_numpy(), hourly.to_numpy(), atol=1e-9)
