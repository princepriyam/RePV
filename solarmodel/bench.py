"""Testbench: run models on weather inputs at sites and score them against measured AC power.

    python -m solarmodel.bench                                  # every site with [measured], reference chain vs RePV
    python -m solarmodel.bench --sites Henderson_1423 LasVegas_34 \\
        --models thesis current current:iam=none current:snow_albedo=null
    python -m solarmodel.bench --study studies/LasVegas_34/inputs.toml

From Python::

    from solarmodel import bench, models
    metrics, hourly = bench.run(["LasVegas_34"], [models.thesis(), models.current(iam="none")])

A series is one model on one weather input (``"thesis|point"``,
``"current|era5:bilinear"``, see :mod:`solarmodel.weather.inputs`) or one model
corrected by MOS (``"current|mos:ridge"``, :mod:`solarmodel.weather.mos`,
leave one year out). Temperature and wind come from one source for all inputs,
so inputs differ only in irradiance.

Scoring, the same for every series on a site:
* measured power is centred on its interval and shifted by one whole-hour lag,
  found against the clear-sky GHI curve (model-free);
* days with an outage, flat-lined or zero output in daylight are dropped;
* daylight = clear-sky GHI above 50 W/m2;
* only hours where every series has a value are scored.
Scores: bias = mean error / mean measured [%], NRMSE = RMSE / mean measured
[%], KGE (Gupta et al. 2009), R2 = 1 - SSE/SST (hourly and daily sums),
energy and specific yield on the scored hours.
:func:`breakdown` splits them by month, local hour, clear-sky index or
temperature.
"""
from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from solarmodel import paths
from solarmodel.model.chain import PVSystem, module_parameters, run_chain
from solarmodel.models import Model, parse
from solarmodel.weather import inputs
from solarmodel.weather.site import hourly_soiling

HALF_HOUR = pd.Timedelta(minutes=30)
# hourly columns that describe the hour, not a series
CONTEXT = ("site", "measured", "ghi", "kc", "temp_air", "year", "month", "local_hour")
DAYLIGHT_WM2 = 50.0                     # clear-sky GHI that counts as daylight


@dataclass
class Site:
    name: str
    config: dict
    system: PVSystem
    start: str                      # scoring window, dates inclusive
    end: str
    capacity_kw: float


def load_site(site: str | Path, start: str | None = None, end: str | None = None) -> Site:
    """A site by name (``config/site/<name>.toml``) or by TOML path; ``start``/``end``
    replace the config's scoring window."""
    path = Path(site) if str(site).endswith(".toml") else paths.CONFIG_DIR / "site" / f"{site}.toml"
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    return Site(name=path.stem, config=cfg, system=PVSystem.from_site_toml(cfg),
                start=start or cfg["data"]["sa_start"], end=end or cfg["data"]["sa_end"],
                capacity_kw=float(cfg["system"]["rated_capacity_kw"]))


def simulate(site: Site, model: Model, weather: pd.DataFrame) -> pd.DataFrame:
    """The chain output for one model on ``weather`` (any regular UTC index). Hourly rows are
    hour-ending means; sub-hourly rows are step centres, run with ``timestamp_label="center"``
    and the step as ``interval_minutes``. A site with ``[snow] removal = "fast"`` runs without
    snow cover (snow is cleared by an unrecorded mechanism or by hand); everything else, snow
    albedo included, is unchanged."""
    step = int(round((weather.index[1] - weather.index[0]) / pd.Timedelta(minutes=1))) if len(weather) > 1 else 60
    if step < 60:
        model = replace(model, options=replace(model.options, timestamp_label="center", interval_minutes=step))
    if site.config.get("snow", {}).get("removal") == "fast" and model.options.snow != "none":
        model = replace(model, options=replace(model.options, snow="none"))
    system = model.system(site.system)
    if model.kind == "homer":
        return _homer(site, system, model, weather)
    soil = hourly_soiling(system.soiling, system.lat, system.lon, site.start, site.end, weather.index)
    return run_chain(weather, system, model.options, soiling_ratio=soil, module_params=module_parameters(system))


def _homer(site: Site, system, model: Model, weather: pd.DataFrame) -> pd.DataFrame:
    """Simple rating model: NOCT cell temperature and PVWatts DC
    on the POA of the chain, without soiling or snow."""
    out = run_chain(weather, system, model.options, soiling_ratio=pd.Series(1.0, index=weather.index),
                    module_params=module_parameters(system))
    c = site.config
    h, mod, los = c["thermal"]["homer"], c["module"], c["losses"]
    poa = out["poa_global"] / 1000.0                                  # kW/m2
    a, eta = mod["alpha_p"], mod["eta_mp_stc"]
    k = (h["tc_noct"] - h["ta_noct"]) * poa / h["poa_noct"]
    tc = (weather["temp_air"] + k * (1 - eta * (1 - a * (25 - h["tc_noct"])))) / (1 + k * eta * a)
    ypv = float(c["system"]["rated_capacity_kw"])
    p = pvlib.pvsystem.pvwatts_dc(poa * 1000.0, tc, pdc0=ypv * los["f_pv"] * los["eta_inv"] * los.get("n_degradation", 1.0) * 1000.0,
                                  gamma_pdc=a, temp_ref=25.0) / 1000.0
    p = p.clip(lower=0.0, upper=min(ypv, system.export_limit_kw))
    return pd.DataFrame({"p_ac_kw": p, "p_dc_kw": p, "clip_kw": 0.0, "poa_global": out["poa_global"],
                         "poa_eff": out["poa_global"], "t_cell": tc, "snow_tx": 1.0, "soiling": 1.0})


def read_measured(m: dict) -> pd.Series:
    """The ``[measured]`` file as AC power [kW] on its own stamps, localised to ``m["tz"]``
    (a stamp that does not exist there, or is ambiguous, is dropped), sorted, first value per stamp."""
    f = paths.PROJECT_ROOT / m["file"]
    raw = pd.read_parquet(f).reset_index() if f.suffix == ".parquet" else pd.read_csv(f, low_memory=False)
    if "id_col" in m:                                   # one file holding several systems
        raw = raw[raw[m["id_col"]] == m["system_id"]]
        t = pd.to_datetime(raw[m["time_col"]], format=m["time_fmt"])
        kw = pd.to_numeric(raw[m["value_col"]], errors="coerce")
    else:
        t = pd.to_datetime(raw[m["time_col"]])
        kw = pd.to_numeric(raw[m["power_col"]], errors="coerce") / {"kw": 1.0, "w": 1000.0}[m["power_unit"].lower()]
    s = pd.Series(kw.to_numpy(), index=pd.DatetimeIndex(t))
    s.index = (s.index.tz_localize(m["tz"], ambiguous="NaT", nonexistent="NaT")
               if s.index.tz is None else s.index.tz_convert(m["tz"]))
    s = s[s.index.notna()]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_measured(site: Site) -> pd.Series:
    """Measured AC [kW], negatives set to 0, centred on its interval, UTC, inside the scoring window."""
    m = site.config["measured"]
    s = read_measured(m).clip(lower=0)
    s = s[(s.index >= pd.Timestamp(site.start, tz=m["tz"]))
          & (s.index < pd.Timestamp(site.end, tz=m["tz"]) + pd.Timedelta(days=1))]
    half = pd.Timedelta(minutes=m["interval_minutes"]) / 2
    s.index = s.index + {"start": half, "end": -half, "center": 0 * half}[m["timestamp_label"]]
    s.index = s.index + pd.Timedelta(hours=m["manual_shift_hours"])
    return s.tz_convert("UTC")


def best_lag(measured: pd.Series, ref: pd.Series, max_lag: int) -> int:
    """Hour shift of the measured series that best matches ``ref`` where ``ref`` > 0."""
    sun = ref > 0
    corr = {lag: measured.shift(lag)[sun].corr(ref[sun]) for lag in range(-max_lag, max_lag + 1)}
    return int(max(corr, key=lambda k: -np.inf if np.isnan(corr[k]) else corr[k]))


def clean_days(measured: pd.Series, ghi_clear: pd.Series, capacity_kw: float, tz: str) -> pd.Series:
    """False on local days with a gap, flat-lined output or three zero hours in
    clear-sky daylight (GHI_clear > 200 W/m2): outages, not weather."""
    day = measured.index.tz_convert(tz).normalize()
    sun = ghi_clear > 200.0
    gap = (measured.isna() & sun).groupby(day).sum() > 0
    flat = (measured.where(sun).groupby(day).max() < 0.02 * capacity_kw) & (ghi_clear.groupby(day).max() > 500.0)
    zeros = ((measured <= 1e-3) & sun).groupby(day).sum() >= 3
    bad = gap | flat | zeros
    return pd.Series(~bad.reindex(day).to_numpy(), index=measured.index)


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    r = np.corrcoef(sim, obs)[0, 1]
    alpha, beta = sim.std() / obs.std(), sim.sum() / obs.sum()
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def r2(sim: np.ndarray, obs: np.ndarray) -> float:
    """Coefficient of determination against measured, 1 - SSE/SST (thesis pipeline.py)."""
    return float(1.0 - ((sim - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def score(sim: pd.Series, obs: pd.Series, tz: str | None = None) -> dict:
    """Scores on the hours given (both series on the same index, no NaN).
    ``r2_daily`` sums the scored hours per local day (``tz``; UTC when None), so
    simulated and measured days cover the same hours."""
    s, o = sim.to_numpy(float), obs.to_numpy(float)
    day = sim.index.tz_convert(tz).date if tz else sim.index.date
    ds, do = sim.groupby(day).sum().to_numpy(float), obs.groupby(day).sum().to_numpy(float)
    return dict(**bias_nrmse(s, o), kge=kge(s, o), r2_hourly=r2(s, o), r2_daily=r2(ds, do), n_days=len(do),
                n_hours=len(o), e_sim_kwh=s.sum(), e_meas_kwh=o.sum())


def bias_nrmse(sim, obs) -> dict:
    """Mean error and RMSE as % of mean measured (arrays, or Series aligned on their index)."""
    e = sim - obs
    return dict(bias_pct=100 * e.mean() / obs.mean(), nrmse_pct=100 * np.sqrt((e ** 2).mean()) / obs.mean())


def align(site: Site) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.Series, int, pd.Series]:
    """The measured side of the scoring protocol on the hour labels: ``labels``, Ineichen clear
    sky at the hour centres (on ``labels``), measured power shifted by ``lag``, ``lag``, and
    ``keep`` (a measured value, daylight, a clean day)."""
    s = site.system
    labels = inputs.hour_labels(site.start, site.end)
    centre = labels - HALF_HOUR                          # hour-ending values describe the hour's centre
    cs = pvlib.location.Location(s.lat, s.lon, altitude=s.alt).get_clearsky(centre, model="ineichen")
    ghi_cs = pd.Series(cs["ghi"].to_numpy(), index=labels)
    m = site.config["measured"]
    meas = pd.Series(load_measured(site).reindex(centre).to_numpy(), index=labels)
    if meas.notna().sum() < 100:
        raise RuntimeError(f"{site.name}: under 100 measured hours on the hour centres; "
                           "check [measured] tz, timestamp_label and interval_minutes")
    lag = best_lag(meas, ghi_cs, m["max_lag_hours"]) if m["auto_lag_correction"] else 0
    meas = meas.shift(lag)
    keep = meas.notna() & (ghi_cs > DAYLIGHT_WM2) & clean_days(meas, ghi_cs, site.capacity_kw, m["tz"])
    return labels, cs.set_axis(labels), meas, lag, keep


def _on_input(site, model, irr, met, snow, labels) -> pd.Series:
    """One model on one irradiance input, as hour-ending AC power [kW] on ``labels``."""
    p = simulate(site, model, inputs.join(irr, met, snow))["p_ac_kw"]
    return p if irr.index.equals(labels) else p.groupby(irr.index.ceil("h")).mean().reindex(labels)


def score_rows(site: Site, hourly: pd.DataFrame, lag: int) -> list[dict]:
    """One row of scores per series column (``model|input``) of ``hourly`` against ``measured``."""
    rows = []
    o = hourly["measured"]
    for name in (c for c in hourly.columns if c not in CONTEXT):
        model, _, spec = name.partition("|")
        p = hourly[name]
        rows.append(dict(site=site.name, model=model, input=spec, lag_h=lag,
                         **score(p, o, tz=site.config["measured"]["tz"]),
                         sy_sim=p.sum() / site.capacity_kw, sy_meas=o.sum() / site.capacity_kw))
    return rows


def run_site(site: Site, models: list[Model], weather=("point",), met="point", grids=None,
             mos=(), mos_grid="era5", benchmarks=()) -> tuple[list[dict], pd.DataFrame]:
    grids = grids or {}
    s = site.system
    station = site.config.get("station", {})
    met_df = inputs.met(met, site.name, s, site.start, site.end, grids, station)
    snow = inputs.snow(s, site.start, site.end, snowfall=any(m.options.snow == "nrel" for m in models),
                       depth=any(m.needs_snow_depth for m in models))
    irr = {spec: inputs.irradiance(spec, site.name, s, site.start, site.end, grids, station) for spec in weather}
    labels, cs, meas, lag, keep = align(site)
    centre = labels - HALF_HOUR
    ghi_cs = cs["ghi"]

    series = {}
    for model in models:
        for spec, x in irr.items():
            series[f"{model.name}|{spec}"] = _on_input(site, model, x, met_df, snow, labels)
        if mos:
            from solarmodel.weather import mos as mos_
            clear = cs[["ghi", "dhi", "dni"]]
            p_clear = _on_input(site, model, clear, met_df, snow, labels)     # the MOS target's denominator
            y = (meas / p_clear).where(keep).clip(0.0, 1.5)
            ds = inputs.grid(grids[mos_grid], site.name, s.alt)
            for spec in mos:
                kind, _, features = spec.partition(":")
                X = mos_.neighbourhood_features(ds, s.lat, s.lon, half=2, features=features or "neighbourhood_advection")
                X.index = X.index.tz_localize("UTC")
                k = mos_.fit_predict_loyo(X.reindex(labels), y, kind=kind)
                series[f"{model.name}|mos:{spec}"] = k * p_clear

    for spec in benchmarks:                              # external tools on their own weather
        from solarmodel.weather import pvgis
        series[f"pvgis|{pvgis.label(spec)}"] = pvgis.power(site.config, site.start, site.end, labels, spec)

    sims = pd.DataFrame(series)
    keep &= sims.notna().all(axis=1)
    context_irr = next(iter(irr.values()))
    ghi = context_irr["ghi"].groupby(context_irr.index.ceil("h")).mean().reindex(labels)
    local = centre.tz_convert(site.config["measured"]["tz"])
    hourly = pd.DataFrame({
        "site": site.name, "measured": meas, "ghi": ghi,
        "kc": (ghi / ghi_cs).where(ghi_cs > 10.0),        # clear-sky index of the first input
        "temp_air": met_df["temp_air"], "year": local.year, "month": local.month, "local_hour": local.hour,
    }, index=labels).join(sims)[keep]
    hourly.index = centre[keep.to_numpy()]
    return score_rows(site, hourly, lag), hourly.rename_axis("time_utc")


def run(sites, models, **kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every model on every site. ``sites``: names, paths or :class:`Site`;
    ``models``: :class:`Model` or spec strings such as ``"current:iam=none"``;
    ``kw``: ``weather``, ``met``, ``grids``, ``mos``, ``mos_grid``, ``benchmarks``
    (see :func:`run_site`)."""
    models = [parse(m) if isinstance(m, str) else m for m in models]
    rows, hourly = [], []
    for s in sites:
        site = s if isinstance(s, Site) else load_site(s)
        r, h = run_site(site, models, **kw)
        rows += r
        hourly.append(h)
        print(f"{site.name}: {len(r)} series on {r[0]['n_hours']} hours", flush=True)
    return pd.DataFrame(rows), pd.concat(hourly)


def breakdown(hourly: pd.DataFrame, by: str, bins: list | None = None) -> pd.DataFrame:
    """Scores per group of hours: ``by`` is a column of ``hourly`` (month,
    local_hour, kc, temp_air, ...), cut into ``bins`` when given. One row per
    (site, group, series) with hours, energies, bias and NRMSE."""
    series = [c for c in hourly.columns if c not in CONTEXT]
    key = pd.cut(hourly[by], bins) if bins else hourly[by]
    rows = []
    for (site, group), h in hourly.groupby([hourly["site"], key], observed=True):
        o = h["measured"].to_numpy(float)
        for name in series:
            if h[name].isna().all():                       # a model this site excluded
                continue
            model, _, spec = name.partition("|")
            p = h[name].to_numpy(float)
            rows.append(dict(site=site, **{by: group}, model=model, input=spec, n_hours=len(o),
                             e_meas_kwh=o.sum(), e_sim_kwh=o.sum() + (p - o).sum(), **bias_nrmse(p, o)))
    return pd.DataFrame(rows)


def run_study(study_file: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run a study TOML and write metrics.csv, hourly.csv.gz and by_<name>.csv
    to ``outputs/<study name>/`` next to it; returns metrics, hourly and that folder.

    Keys: ``sites``, ``models``; optional ``start``/``end`` (replace the sites'
    windows), ``weather`` (irradiance inputs, default ["point"]), ``met``,
    ``[grids]`` (name -> folder), ``mos`` and ``mos_grid``, ``benchmarks``
    (``["pvgis"]``: PVGIS-ERA5 output, series ``pvgis|PVGIS-ERA5``), ``[breakdowns]``.
    A ``sites`` entry is a name (or .toml path) or a table
    ``{ name = "...", start = "...", end = "...", exclude = ["<model spec>", ...] }``:
    its window replaces the study's, and ``exclude`` lists models the site cannot run
    (e.g. ``inverter=sandia`` without an ``[inverter].cec`` entry).
    """
    study_file = Path(study_file)
    with open(study_file, "rb") as f:
        st = tomllib.load(f)

    def _site(entry):
        e = entry if isinstance(entry, dict) else {"name": entry}
        n = e["name"]
        bad = set(e.get("exclude", [])) - set(st["models"])
        if bad:
            raise ValueError(f"{n}: exclude names models not in the study: {sorted(bad)}")
        return (load_site(study_file.parent / n if n.endswith(".toml") else n,
                          e.get("start", st.get("start")), e.get("end", st.get("end"))),
                [m for m in st["models"] if m not in e.get("exclude", [])])

    kw = {k: st[k] for k in ("weather", "met", "grids", "mos", "mos_grid", "benchmarks") if k in st}
    parts = [run([site], models, **kw) for site, models in map(_site, st["sites"])]
    metrics = pd.concat([m for m, _ in parts], ignore_index=True)
    hourly = pd.concat([h for _, h in parts])
    out = study_file.parent / "outputs" / study_file.stem
    write(out, metrics, hourly, st.get("breakdowns", {}))
    return metrics, hourly, out


def write(out: Path, metrics: pd.DataFrame, hourly: pd.DataFrame, breakdowns: dict | None = None) -> None:
    """``metrics.csv``, ``hourly.csv.gz`` and ``by_<name>.csv`` per ``{name: {by, bins}}`` in ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    hourly.to_csv(out / "hourly.csv.gz")
    for name, b in (breakdowns or {}).items():
        breakdown(hourly, b["by"], b.get("bins")).to_csv(out / f"by_{name}.csv", index=False)


def measured_sites() -> list[str]:
    out = []
    for f in sorted((paths.CONFIG_DIR / "site").glob("*.toml")):
        with open(f, "rb") as fh:
            if "measured" in tomllib.load(fh):
                out.append(f.stem)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="+", default=None, help="site names (default: every site with [measured])")
    ap.add_argument("--models", nargs="+", default=["thesis", "current"],
                    help='model specs, e.g. thesis current "current:iam=none"')
    ap.add_argument("--out", default=str(paths.RESULTS_DIR / "bench"))
    ap.add_argument("--study", help="a study TOML; replaces the other options")
    a = ap.parse_args(argv)
    if a.study:
        metrics, _, out = run_study(a.study)
    else:
        metrics, hourly = run(a.sites or measured_sites(), a.models)
        out = Path(a.out)
        write(out, metrics, hourly)
    with pd.option_context("display.width", 200, "display.precision", 2, "display.max_colwidth", 60):
        print(metrics.set_index(["site", "model", "input"])[["bias_pct", "nrmse_pct", "kge", "n_hours"]])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
