"""Run RePV offline on the Las Vegas 34 test year (2011) shipped in tests/data/bench.

    python examples/run_repv.py

Writes examples/repv_LasVegas_34_2011.csv (hourly AC power and intermediate series, UTC
hour-ending labels) and prints the annual energy. Uses the cached ERA5 weather of the test
data, so no network access is needed. For a configured site with downloads, see README.md.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solarmodel import bench, models, paths  # noqa: E402
from solarmodel.weather import inputs  # noqa: E402

DATA = ROOT / "tests" / "data" / "bench"


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    paths.CACHE_DIR = DATA                                 # the cached test weather
    site = bench.load_site(DATA / "LasVegas_34.toml")
    m = models.repv()
    w = inputs.weather("point", "point", site.name, site.system, site.start, site.end, {}, m)
    out = bench.simulate(site, m, w)
    f = Path(__file__).parent / "repv_LasVegas_34_2011.csv"
    out.to_csv(f, float_format="%.6g")
    print(f"{site.name} {site.start[:4]}: {out['p_ac_kw'].sum() / 1000:.1f} MWh AC "
          f"({out['p_ac_kw'].sum() / site.capacity_kw:.0f} kWh/kWp) -> {f}")


if __name__ == "__main__":
    main()
