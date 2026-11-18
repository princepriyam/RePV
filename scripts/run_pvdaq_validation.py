"""Run the model comparison and validation on every PVDAQ site, site by site.

    python scripts/run_pvdaq_validation.py                    # all PVDAQ sites; done sites are skipped
    python scripts/run_pvdaq_validation.py --sites Golden_50 CherryHill_1201 --redo
    python scripts/run_pvdaq_validation.py --combine-only     # rebuild the combined tables
    python scripts/run_pvdaq_validation.py --dry-run          # list sites, windows and inputs only

Settings (models, irradiance inputs, temperature/wind source, benchmarks, breakdowns) are in
studies/PVDAQ_validation/validation.toml. Sites are the config/site/*.toml files whose
[measured] file is a PVDAQ hourly parquet (written by pvdaq_site_configs.py).

Per site:
  1. the measured parquet is checked again (pvdaq_verify_parquet.check); a site with a
     structural failure is skipped, a power-only warning (DC rating) is recorded and run
  2. bench.run_site scores every model on every input against the measured data on the same
     clean daylight hours: the study's inputs, plus "midc" where the site has a [station]; the
     PVGIS benchmark where PVGIS-ERA5 covers the scoring year
  3. outputs/pvdaq_validation/sites/<site>/ gets hourly.parquet, metrics.parquet, run.json
A failing site is logged in errors.json and the run goes on.
Then: metrics.parquet (+ metrics.csv), by_<breakdown>.parquet (+ .csv) and summary.csv for
all sites. Weather comes only from the parquet caches (cache/weather, cache/pvgis,
cache/horizon); legacy CSV caches are converted and deleted first.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import tomllib
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarmodel import paths  # noqa: E402

ROOT = paths.PROJECT_ROOT                  # the root bench resolves configs and measured files against
STUDY = ROOT / "studies" / "PVDAQ_validation" / "validation.toml"
OUT = ROOT / "outputs" / "pvdaq_validation"
PVGIS_YEARS = (2005, 2023)                 # PVGIS-ERA5 coverage in PVGIS 5.3
STRUCTURAL = ("schema", "index", "consistency", "dark_hours", "timestamps", "bench")


def _verify():
    spec = importlib.util.spec_from_file_location("verify", Path(__file__).parent / "pvdaq_verify_parquet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pvdaq_sites() -> list[tuple[str, int]]:
    out = []
    for f in sorted((paths.CONFIG_DIR / "site").glob("*.toml")):
        with open(f, "rb") as fh:
            m = tomllib.load(fh).get("measured", {})
        p = str(m.get("file", ""))
        if p.startswith("data/validation/PVDAQ/") and p.endswith("_hourly.parquet"):
            out.append((f.stem, int(Path(p).parent.name)))
    return out


def plan(site, st: dict) -> dict:
    weather = list(st.get("weather", ["point"]))
    if site.config.get("station", {}).get("midc") and "midc" not in weather:
        weather.append("midc")
    y0, y1 = int(site.start[:4]), int(site.end[:4])
    bench_ok = PVGIS_YEARS[0] <= y0 and y1 <= PVGIS_YEARS[1]
    return dict(weather=weather, met=st.get("met", "point"),
                benchmarks=list(st.get("benchmarks", [])) if bench_ok else [],
                note="" if bench_ok or not st.get("benchmarks") else f"PVGIS-ERA5 covers {PVGIS_YEARS[0]}-{PVGIS_YEARS[1]} only: no PVGIS benchmark")


def run_one(name: str, sid: int, st: dict, redo: bool, verify) -> str:
    from solarmodel import bench
    from solarmodel.models import parse
    d = OUT / "sites" / name
    if not redo and (d / "metrics.parquet").exists() and (d / "hourly.parquet").exists():
        return "done earlier, skipped"
    chk = verify.check(sid)
    bad = [k for k in STRUCTURAL if str(chk.get(k, "")).startswith("FAIL")]
    if bad:
        raise RuntimeError(f"measured parquet fails {bad}: " + "; ".join(str(chk[k]) for k in bad))
    site = bench.load_site(name)
    p = plan(site, st)
    specs = [m for m in st["models"] if m not in st.get("exclude", {}).get(name, [])]
    if "cec" not in site.config.get("inverter", {}):          # the Sandia variant needs a CEC inverter entry
        specs = [m for m in specs if "inverter=sandia" not in m]
    models = [parse(m) for m in specs]
    t0 = time.time()
    rows, hourly = bench.run_site(site, models, weather=p["weather"], met=p["met"], benchmarks=p["benchmarks"])
    d.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(rows).assign(system_id=sid)
    hourly.to_parquet(d / "hourly.parquet")
    m.to_parquet(d / "metrics.parquet", index=False)
    (d / "run.json").write_text(json.dumps(dict(
        site=name, system_id=sid, window=[site.start, site.end], **p, models=[x.name for x in models],
        measured_check=chk.get("verdict"), seconds=round(time.time() - t0, 1),
        finished=pd.Timestamp.now().isoformat(timespec="seconds")), indent=1))
    warn = "" if chk.get("verdict") == "PASS" else f"; measured check: {chk.get('verdict')}"
    return f"{len(m)} series on {int(m.n_hours.max())} hours ({', '.join(p['weather'])}" + \
           (f"; {p['note']}" if p["note"] else "") + f"){warn}"


def combine(st: dict) -> None:
    from solarmodel import bench
    excl = st.get("excluded_sites", {})
    sites = sorted(p for p in (OUT / "sites").glob("*") if (p / "metrics.parquet").exists() and p.name not in excl)
    if not sites:
        print("nothing to combine")
        return
    metrics = pd.concat([pd.read_parquet(p / "metrics.parquet") for p in sites], ignore_index=True)
    metrics.to_parquet(OUT / "metrics.parquet", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    for bname, b in st.get("breakdowns", {}).items():
        parts = [bench.breakdown(pd.read_parquet(p / "hourly.parquet"), b["by"], b.get("bins")) for p in sites]
        t = pd.concat(parts, ignore_index=True)
        t.to_parquet(OUT / f"by_{bname}.parquet", index=False)
        t.to_csv(OUT / f"by_{bname}.csv", index=False)
    cols = [c for c in ("bias_pct", "nrmse_pct", "kge", "r2_hourly", "r2_daily") if c in metrics]
    s = metrics.groupby(["model", "input"])[cols].median().round(2)
    s["sites"] = metrics.groupby(["model", "input"]).site.nunique()
    s.to_csv(OUT / "summary.csv")
    with pd.option_context("display.width", 200):
        print(f"\nmedian over sites ({len(sites)} sites; excluded: {', '.join(sorted(excl)) or 'none'}):\n{s}")
    print(f"-> {OUT}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="+", help="site names (default: every PVDAQ site config)")
    ap.add_argument("--redo", action="store_true", help="run again even if the site's outputs exist")
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--study", default=str(STUDY))
    a = ap.parse_args(argv)
    with open(a.study, "rb") as f:
        st = tomllib.load(f)
    OUT.mkdir(parents=True, exist_ok=True)

    from solarmodel.weather import cache
    moved = [] if a.dry_run else cache.migrate()
    if moved:
        print(f"weather cache: {len(moved)} CSV files converted to parquet and deleted")

    if not a.combine_only:
        from solarmodel import bench
        verify = _verify()
        excl = st.get("excluded_sites", {})
        allsites = [(n, s) for n, s in pvdaq_sites() if not a.sites or n in a.sites]
        missing = set(a.sites or []) - {n for n, _ in allsites}
        todo = [(n, s) for n, s in allsites if n not in excl]
        for n, _ in allsites:
            if n in excl:
                print(f"  {n:<24} excluded: {excl[n]}")
        if missing:
            raise SystemExit(f"not PVDAQ site configs: {sorted(missing)}")
        ef = OUT / "errors.json"
        errors = json.loads(ef.read_text()) if ef.exists() else {}
        print(f"{len(todo)} sites; models: {', '.join(st['models'])}", flush=True)
        for name, sid in todo:
            if a.dry_run:
                site = bench.load_site(name)
                p = plan(site, st)
                print(f"  {name:<24} {site.start}..{site.end}  inputs {p['weather']}, met {p['met']}, "
                      f"benchmarks {p['benchmarks'] or '-'} {p['note']}")
                continue
            try:
                msg = run_one(name, sid, st, a.redo, verify)
                errors.pop(name, None)
            except Exception as e:
                msg = f"ERROR {type(e).__name__}: {e}"
                errors[name] = dict(error=msg, traceback=traceback.format_exc(limit=6),
                                    at=pd.Timestamp.now().isoformat(timespec="seconds"))
            ef.write_text(json.dumps(errors, indent=1))
            print(f"  {name:<24} {msg}", flush=True)
        if a.dry_run:
            return
        if errors:
            print(f"{len(errors)} sites failed: {sorted(errors)} (details in {ef})")
    combine(st)


if __name__ == "__main__":
    main()
