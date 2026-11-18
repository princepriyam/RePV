"""How often the ground-snow-depth condition and reset change the RePV snow state.

    python scripts/snow_depth_binding.py          # every independent assessment site with snowfall

For each site and its scored year, RePV is run on ERA5 with and without the depth
condition and reset (snow_use_depth), soiling off, and the script counts
  events_suppressed  hours whose trailing 6 h snowfall exceeds 1 cm while the
                     ERA5-Land depth is below 1 cm (the condition removes an event)
  hours_changed      hours in which the snow transmittance differs between the two runs
  daylight_changed   of those, hours with plane-of-array irradiance above zero
  energy_change_pct  change in annual AC energy caused by the condition, in percent
Reads the weather cache only (cache/weather, cache/horizon). Section 3.5 of the article.
"""
from __future__ import annotations

import sys
import tomllib
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarmodel import bench, models, paths  # noqa: E402
from solarmodel.model.chain import run_chain  # noqa: E402
from solarmodel.weather import inputs  # noqa: E402

CALIBRATED = {"CherryHill_1201", "Chester_1204", "Gaithersburg_4902", "Linthicum_1200"}


def sites() -> list[str]:
    st = tomllib.loads((paths.PROJECT_ROOT / "studies" / "PVDAQ_validation" / "validation.toml").read_text())
    excluded = set(st.get("excluded_sites", {}))
    out = []
    for f in sorted((paths.CONFIG_DIR / "site").glob("*.toml")):
        m = tomllib.loads(f.read_text()).get("measured", {})
        if str(m.get("file", "")).startswith("data/validation/PVDAQ/") and f.stem not in excluded | CALIBRATED:
            out.append(f.stem)
    return out


def one(name: str) -> dict | None:
    site = bench.load_site(name)
    with_depth = models.repv()
    without = models.repv(snow_use_depth=False)
    w = inputs.weather("point", "point", site.name, site.system, site.start, site.end, {}, with_depth)
    if w["snowfall_cm"].sum() == 0:
        return None
    one_ = pd.Series(1.0, index=w.index)
    a = run_chain(w, with_depth.system(site.system), with_depth.options, soiling_ratio=one_)
    b = run_chain(w, without.system(site.system), without.options, soiling_ratio=one_)
    roll = w["snowfall_cm"].rolling(6, min_periods=1).sum()
    changed = (a["snow_tx"] - b["snow_tx"]).abs() > 1e-12
    return dict(site=name, events_suppressed=int(((roll > 1.0) & (w["snow_depth_cm"] < 1.0)).sum()),
                hours_changed=int(changed.sum()), daylight_changed=int((changed & (a["poa_global"] > 0)).sum()),
                energy_change_pct=100 * (a["p_ac_kw"].sum() / b["p_ac_kw"].sum() - 1))


def main():
    warnings.filterwarnings("ignore")
    rows = []
    for s in sites():
        try:
            r = one(s)
        except OSError as e:                       # weather not cached and no network
            print(f"{s}: skipped ({type(e).__name__})")
            continue
        if r:
            rows.append(r)
    df = pd.DataFrame(rows).set_index("site")
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n{len(df)} sites with snowfall; events suppressed: {df.events_suppressed.sum()}; "
          f"max hours changed at one site: {df.hours_changed.max()}; "
          f"max |energy change|: {df.energy_change_pct.abs().max():.3f} %")


if __name__ == "__main__":
    main()
