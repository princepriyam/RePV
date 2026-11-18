"""Assemble the archive of RePV for Zenodo: code, paper, site files and the data of the article.

    python scripts/make_release.py               # release/repv-<version>/
    python scripts/make_release.py --zip         # also release/repv-<version>.zip (upload to Zenodo)
    python scripts/make_release.py --dry-run     # list what would be copied, with sizes

The git repository holds the code and the paper; the weather cache, the measured series and
the per-site outputs stay out of git (.gitignore) and travel in this archive. Only what the
article needs is copied, so local credentials, scratch files and basemap tiles are never
published. Included:
  code          solarmodel/, tests/, scripts/ (no .bak), examples/, pyproject.toml,
                requirements-lock.txt, README.md, LICENSE, CITATION.cff, CHANGELOG.md
  study         studies/PVDAQ_validation/validation.toml, studies/LasVegas_34/fit_thermal.py
  sites         config/site/<PVDAQ site>.toml (assessment and excluded sites),
                config/pvdaq_overrides.json, config/secrets.example.toml
  data          data/validation/PVDAQ/<id>/<id>_hourly.parquet, _metadata.json, _report.md;
                cache/weather, cache/horizon, cache/pvgis files of those sites and their windows;
                outputs/pvdaq_inventory, outputs/pvdaq_validation
  paper         latex/methodsx sources, generators and PDFs; latex/methodsx_figures scripts,
                data and figures
The release's own .gitignore keeps the data directories out of a git repository, so that
`git init && git add . && git commit` gives a light code repository for GitHub, while the zip
with the data is the archive for Zenodo. Standard library only (Python 3.10+).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE_GLOBS = ["solarmodel/**/*.py", "solarmodel/README.md", "tests/**/*.py", "tests/data/**/*",
              "scripts/*.py", "examples/*.py", "pyproject.toml", "requirements-lock.txt", "README.md",
              "LICENSE", "CITATION.cff", "CHANGELOG.md",
              "studies/PVDAQ_validation/validation.toml", "studies/LasVegas_34/fit_thermal.py",
              ".github/workflows/*.yml",
              "config/pvdaq_overrides.json", "config/secrets.example.toml"]
PAPER = ["latex/methodsx/main.tex", "latex/methodsx/supplement.tex", "latex/methodsx/macros.tex",
         "latex/methodsx/numbers.tex", "latex/methodsx/references.bib", "latex/methodsx/build.sh",
         "latex/methodsx/worked_example.tex", "latex/methodsx/supplementary.md",
         "latex/methodsx/make_validation_tables.py", "latex/methodsx/make_worked_example.py",
         "latex/methodsx/make_repo_supplement.py", "latex/methodsx/make_overleaf.py", "latex/methodsx/main.pdf", "latex/methodsx/supplement.pdf"]
PAPER_GLOBS = ["latex/methodsx/val_*.tex", "latex/methodsx_figures/*.R", "latex/methodsx_figures/*.py",
               "latex/methodsx_figures/*.sh", "latex/methodsx_figures/data/**/*",
               "latex/methodsx_figures/out/fig_validation.pdf", "latex/methodsx_figures/out/fig_sites_map.pdf"]
# scratch files, and basemap tiles (provider terms do not allow redistribution; fig_sites_map.R downloads them)
SKIP = re.compile(r"(__pycache__|\.pyc$|\.bak$|\.DS_Store$|/\.pre_|\.pre_pvdaq$|\.pre_metadata$|/tiles/)")
GITIGNORE = """# data of the article: in the Zenodo archive, not in the code repository
/cache/
/data/
latex/methodsx_figures/data/tiles/
outputs/pvdaq_validation/sites/
outputs/pvdaq_validation/*.parquet
config/secrets.toml
__pycache__/
*.pyc
.pytest_cache/
latex/methodsx/build/
latex/methodsx_overleaf/
latex/methodsx_overleaf.zip
examples/repv_*.csv
*.egg-info/
.ruff_cache/
.venv/
.env
.DS_Store
release/
latex/methodsx/*.aux
latex/methodsx/*.bbl
latex/methodsx/*.blg
latex/methodsx/*.log
latex/methodsx/*.out
latex/methodsx/*.fls
latex/methodsx/*.fdb_latexmk
latex/methodsx/*.synctex.gz
studies/*/outputs/
"""


def version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)
    return m.group(1)


def pvdaq_sites() -> list[tuple[str, str, str, str]]:
    """(name, system id, lat, lon) of every site file whose measured series is a PVDAQ extract."""
    out = []
    for f in sorted((ROOT / "config" / "site").glob("*.toml")):
        t = f.read_text()
        m = re.search(r'^file\s*=\s*"data/validation/PVDAQ/(\d+)/', t, re.M)
        if not m:
            continue
        lat = re.search(r"^lat\s*=\s*([-0-9.]+)", t, re.M).group(1)
        lon = re.search(r"^lon\s*=\s*([-0-9.]+)", t, re.M).group(1)
        out.append((f.stem, m.group(1), lat, lon))
    return out


def scored_years() -> dict[str, set[int]]:
    """Calendar years of each site's assessment window, from the run records."""
    out = {}
    for f in (ROOT / "outputs" / "pvdaq_validation" / "sites").glob("*/run.json"):
        m = re.search(r'"window":\s*\[\s*"(\d{4})-\d\d-\d\d",\s*"(\d{4})-', f.read_text())
        if m:
            out[f.parent.name] = set(range(int(m.group(1)), int(m.group(2)) + 1))
    return out


def station_years() -> set[tuple[str, str]]:
    """(MIDC station, YYYY) of the assessment windows, from the run records."""
    out = set()
    for f in (ROOT / "outputs" / "pvdaq_validation" / "sites").glob("*/run.json"):
        t = f.read_text()
        if '"midc"' not in t:
            continue
        site = (ROOT / "config" / "site" / f"{f.parent.name}.toml").read_text()
        st = re.search(r'^midc\s*=\s*"([^"]+)"', site, re.M)
        yrs = re.findall(r'"(\d{4})-\d\d-\d\d"', t)
        if st:
            for y in set(yrs):
                out.add((st.group(1), y))
    return out


def files() -> list[Path]:
    sel: set[Path] = set()
    for g in CODE_GLOBS + PAPER_GLOBS:
        sel |= {p for p in ROOT.glob(g) if p.is_file()}
    sel |= {ROOT / p for p in PAPER if (ROOT / p).exists()}
    sites = pvdaq_sites()
    years = scored_years()
    for name, sid, lat, lon in sites:
        sel.add(ROOT / "config" / "site" / f"{name}.toml")
        d = ROOT / "data" / "validation" / "PVDAQ" / sid
        sel |= {d / f"{sid}{s}" for s in ("_hourly.parquet", "_metadata.json", "_report.md") if (d / f"{sid}{s}").exists()}
        keys = {f"__{lat}__{lon}", f"__{float(lat):.4f}__{float(lon):.4f}"}
        want = {y + k for y in years.get(name, set()) for k in (-1, 0, 1)}   # UTC shifts reach the neighbours
        for sub in ("weather", "horizon", "pvgis"):
            for p in (ROOT / "cache" / sub).glob("*"):
                if not any(k in p.name for k in keys):
                    continue
                file_years = {int(y) for y in re.findall(r"__(\d{4})(?:-\d\d-\d\d)?", p.name)}
                if sub == "horizon" or file_years & want:
                    sel.add(p)
    for st, y in station_years():
        sel |= set((ROOT / "cache" / "weather").glob(f"midc__{st}__{y}-*.parquet"))
        sel |= set((ROOT / "cache" / "weather").glob(f"midc__{st}__{int(y) + 1}-01.parquet"))
    for sub in ("pvdaq_inventory", "pvdaq_validation"):
        sel |= {p for p in (ROOT / "outputs" / sub).rglob("*") if p.is_file()}
    return sorted(p for p in sel if not SKIP.search(str(p)) and p.name != "secrets.toml")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zip", action="store_true", help="also write the zip archive for Zenodo")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    fs = files()
    size = sum(p.stat().st_size for p in fs)
    name = f"repv-{version()}"
    print(f"{len(fs)} files, {size / 1e6:.0f} MB")
    if a.dry_run:
        for p in fs:
            print(f"{p.stat().st_size / 1e6:9.2f} MB  {p.relative_to(ROOT)}")
        return
    dest = ROOT / "release" / name
    if dest.exists():
        try:
            shutil.rmtree(dest)
        except PermissionError:                      # a folder where deleting is not allowed
            print(f"note: {dest} could not be cleared; its files are overwritten in place")
    for p in fs:
        q = dest / p.relative_to(ROOT)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    (dest / ".gitignore").write_text(GITIGNORE)
    print(f"-> {dest}")
    if a.zip:
        z = ROOT / "release" / f"{name}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in [dest / q.relative_to(ROOT) for q in fs] + [dest / ".gitignore"]:   # exactly this release
                kind = zipfile.ZIP_STORED if p.suffix in (".parquet", ".pdf", ".zip") else zipfile.ZIP_DEFLATED
                zf.write(p, Path(name) / p.relative_to(dest), compress_type=kind)
        print(f"-> {z} ({z.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
