"""Extract, correct and store every screened PVDAQ system, one after the other.

    python scripts/pvdaq_extract_all.py                     # tiers A and B of screen.csv
    python scripts/pvdaq_extract_all.py --tiers A B M G     # also multi-orientation and missing geometry
    python scripts/pvdaq_extract_all.py --ids 1422 1403     # selected systems
    python scripts/pvdaq_extract_all.py --redo              # ignore systems already done

Per system it calls pvdaq_extract.process: metadata and data checks, raw AC cache per year
(parquet), sample-level corrections (above 1.3 x DC or below -5 % of DC -> NaN; standby
negatives -> 0), hourly means written as CSV and parquet, the parquet read back and compared
with both the in-memory frame and the CSV, and the CSV deleted only if they match.
One failing system does not stop the run.

Writes outputs/pvdaq_inventory/extract_summary.csv: verdict, best year, years with
at least 80 % scorable daylight hours, timestamp check, DC rating and its source,
warnings and failures, file status.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarmodel import paths  # noqa: E402

ROOT = paths.PROJECT_ROOT                  # the root bench resolves configs and measured files against
OUT = ROOT / "outputs" / "pvdaq_inventory"
_spec = importlib.util.spec_from_file_location("ext", Path(__file__).parent / "pvdaq_extract.py")
ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ext)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tiers", nargs="+", default=["A", "B"], help="tiers of screen.csv to extract")
    ap.add_argument("--ids", nargs="+", type=int, help="system ids (instead of tiers)")
    ap.add_argument("--redo", action="store_true", help="extract again even if the hourly parquet exists")
    ap.add_argument("--drop-raw", action="store_true", help="delete each raw cache once its hourly parquet is verified")
    ap.add_argument("--from-cache", action="store_true", help="rebuild from the cached raw files, no OEDI access")
    a = ap.parse_args(argv)

    if a.ids:
        ids = a.ids
    else:
        sc = pd.read_csv(OUT / "screen.csv")
        keep = sc.status.str.match("|".join(rf"tier {t}\b" for t in a.tiers))
        ids = sc.loc[keep, "system_id"].astype(int).tolist()
    print(f"{len(ids)} systems: {ids}", flush=True)

    summ_f = OUT / "extract_summary.csv"
    done = pd.read_csv(summ_f) if summ_f.exists() else pd.DataFrame(columns=["system_id"])
    rows = done.to_dict("records")              # --redo replaces only the systems it runs
    have = {int(r["system_id"]) for r in rows}
    for sid in ids:
        pq = ROOT / "data" / "validation" / "PVDAQ" / str(sid) / f"{sid}_hourly.parquet"
        if sid in have and pq.exists() and not a.redo:
            print(f"{sid}: done, skipped", flush=True)
            continue
        rows = [r for r in rows if int(r["system_id"]) != sid]
        try:
            rows.append(ext.process(sid, drop_raw=a.drop_raw, from_cache=a.from_cache))
        except Exception as e:
            traceback.print_exc()
            rows.append(dict(system_id=sid, verdict="ERROR", warn_fail=f"{type(e).__name__}: {e}"))
        pd.DataFrame(rows).to_csv(summ_f, index=False)          # saved after every system

    s = pd.DataFrame(rows)
    print(s["verdict"].value_counts().to_string())
    cols = [c for c in ("system_id", "name", "verdict", "best_year", "best_scorable_pct", "years_80pct", "file") if c in s]
    print(s[cols].to_string(index=False))
    print(f"-> {summ_f}")


if __name__ == "__main__":
    main()
