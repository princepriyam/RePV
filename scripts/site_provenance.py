"""Write the [provenance] table of the PVDAQ assessment site files.

    python scripts/site_provenance.py            # annotate every PVDAQ site file in config/site
    python scripts/site_provenance.py --check    # report missing classes only, write nothing

Each model parameter of a site file gets one of the six source classes of
:mod:`solarmodel.schema` (measured, datasheet, published, metadata, output, assumed).
The class follows from the parameter and, where the same parameter can come from
different sources, from the comment written next to its value by
pvdaq_site_configs.py or by hand (for example U_c: "NOCT" = datasheet, "fit ... sensor"
= measured, PVsyst mounting class = published). The table replaces any earlier one.
pvdaq_site_configs.py calls :func:`annotate` after writing a file.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarmodel import paths  # noqa: E402
from solarmodel.schema import missing_provenance, parameter_keys, validate_site  # noqa: E402

HEADER = "# ── Provenance: source class of every model parameter (scripts/site_provenance.py) ──"


def comments(text: str) -> dict[str, str]:
    """``{"section.key": inline comment}`` of a site file (comments of [thermal.x] under "thermal.x")."""
    out, sec = {}, ""
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"^\[([^\]]+)\]", s)
        if m:
            sec = m.group(1).strip()
            continue
        m = re.match(r'^"?([A-Za-z0-9_]+)"?\s*=\s*(.*)$', s)
        if m and sec:
            c = m.group(2).split("#", 1)
            out[f"{sec}.{m.group(1)}"] = c[1].strip().lower() if len(c) > 1 else ""
    return out


def classify(key: str, comment: str, cls_so_far: dict) -> str:
    sec, k = key.rsplit(".", 1)
    assumed = "assumed" in comment or "assumption" in comment
    if sec == "location":
        return "metadata"
    if sec == "array":
        return "assumed" if k in ("albedo", "snow_tilt") or assumed else "metadata"
    if sec == "system":
        if k == "np_arr" and cls_so_far.get("system.ns_arr") == "assumed":
            return "assumed"                           # the string split is assumed as a whole
        return "assumed" if assumed else "metadata"
    if sec == "module":
        if k == "eg_ref":
            return "published"
        if k == "bifaciality":
            return "assumed"
        return "assumed" if assumed else "datasheet"
    if sec == "iam":
        return "published"
    if sec == "thermal.pvsyst":
        if k == "u_c":
            if "noct" in comment:
                return "datasheet"
            if "fit" in comment and "sensor" in comment:
                return "measured"
            if any(w in comment for w in ("pvsyst", "free-standing", "mounting", "roof", "integrated", "insulated")):
                return "published"
            return "assumed"
        if k == "eta_module":
            return "datasheet"
        return "published"                                # u_v = 0 and alpha_abs = 0.9: PVsyst convention
    if sec == "thermal.homer":
        if k == "tc_noct":
            return "assumed" if assumed or "no noct" in comment else "datasheet"
        return "published"
    if sec == "thermal.faiman":
        return "published"
    if sec == "losses":
        if k == "n_degradation":
            return "published"                            # Jordan and Kurtz rate; age from the metadata (comment)
        return "assumed"
    if sec == "soiling":
        return "assumed" if assumed else "published"     # Kimber et al. defaults
    if sec in ("inverter", "layout", "tracker"):
        return "assumed" if assumed else "metadata"
    if sec == "snow":
        return "output"                                   # fast removal: set after comparison with measured output
    return "assumed"


def table(path: Path) -> dict[str, str]:
    text = path.read_text()
    site = tomllib.loads(text)
    site.pop("provenance", None)
    com = comments(text)
    out: dict[str, str] = {}
    for key in parameter_keys(site):
        out[key] = classify(key, com.get(key, ""), out)
    return out


def annotate(path: Path) -> dict[str, str]:
    """Replace the [provenance] table of one site file; return it."""
    text = path.read_text()
    i = text.find(HEADER)
    if i < 0:
        m = re.search(r"^\[provenance\]", text, re.M)
        i = m.start() if m else -1
    body = (text[:i] if i >= 0 else text).rstrip() + "\n"
    prov = table(path)
    w = max(len(k) for k in prov) + 2
    lines = [HEADER, "# measured | datasheet | published | metadata | output | assumed (solarmodel/schema.py)",
             "[provenance]"]
    lines += [f'{chr(34) + k + chr(34):<{w}} = "{c}"' for k, c in prov.items()]
    path.write_text(body + "\n" + "\n".join(lines) + "\n")
    site = tomllib.loads(path.read_text())
    validate_site(site)
    assert not missing_provenance(site), missing_provenance(site)
    return prov


def pvdaq_site_files() -> list[Path]:
    out = []
    for f in sorted((paths.CONFIG_DIR / "site").glob("*.toml")):
        m = tomllib.loads(f.read_text()).get("measured", {})
        if str(m.get("file", "")).startswith("data/validation/PVDAQ/"):
            out.append(f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="report only")
    a = ap.parse_args(argv)
    for f in pvdaq_site_files():
        if a.check:
            miss = missing_provenance(tomllib.loads(f.read_text()))
            print(f"{f.stem:22s} {'complete' if not miss else 'missing: ' + ', '.join(miss)}")
        else:
            prov = annotate(f)
            n = {c: list(prov.values()).count(c) for c in sorted(set(prov.values()))}
            print(f"{f.stem:22s} {n}")


if __name__ == "__main__":
    main()
