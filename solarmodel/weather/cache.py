"""Parquet caches for downloaded inputs (PVGIS series and components, terrain horizons).

The caches used to be CSV. ``read`` converts a legacy CSV next to the parquet path on first
use: it writes the parquet, reads it back, checks it equals the frame read from the CSV, and
only then deletes the CSV. ``migrate`` does the same for every CSV left in the cache folders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from solarmodel import paths


def _legacy_reader(fp: Path) -> Callable[[Path], pd.DataFrame]:
    if fp.parent.name == "horizon":
        def r(p):
            d = pd.read_csv(p, index_col=0)
            d.index = d.index.astype(float)
            return d
        return r
    return lambda p: pd.read_csv(p, index_col=0, parse_dates=True)


def write(fp: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Write ``df`` to ``fp`` (parquet) and return what reads back."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fp)
    back = pd.read_parquet(fp)
    pd.testing.assert_frame_equal(back, df, check_freq=False)
    return back


def read(fp: Path, legacy: Callable[[Path], pd.DataFrame] | None = None) -> pd.DataFrame | None:
    """The cached frame at ``fp`` (parquet), converting a legacy ``fp.with_suffix('.csv')``
    first if there is one; None if neither exists."""
    fp = Path(fp)
    if fp.exists():
        return pd.read_parquet(fp)
    csv = fp.with_suffix(".csv")
    if not csv.exists():
        return None
    df = (legacy or _legacy_reader(fp))(csv)
    back = write(fp, df)                         # raises if the parquet differs: the CSV is kept
    csv.unlink()
    print(f"[cache] {csv.name} -> parquet, CSV deleted", flush=True)
    return back


def migrate(folders=("pvgis", "horizon")) -> list[str]:
    """Convert every CSV in ``cache/<folder>`` to parquet (checked, then deleted)."""
    done = []
    for name in folders:
        d = paths.CACHE_DIR / name
        for csv in sorted(d.glob("*.csv")) if d.exists() else []:
            pq = csv.with_suffix(".parquet")
            if pq.exists():                          # both present: delete the CSV only if they agree
                try:
                    pd.testing.assert_frame_equal(pd.read_parquet(pq), _legacy_reader(pq)(csv), check_freq=False)
                except AssertionError:
                    done.append(f"{name}/{csv.name}: differs from its parquet, kept")
                    continue
                csv.unlink()
            else:
                read(pq)
            done.append(f"{name}/{csv.name}")
    return done
