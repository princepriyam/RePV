"""Numbers and tables of the MethodsX assessment, from outputs/pvdaq_validation.

    python make_validation_tables.py              # bootstrap (2000 replicates) and every .tex file
    python make_validation_tables.py --reps 200   # quick check

Reads   outputs/pvdaq_validation/sites/<site>/hourly.parquet   (scored hours, every series)
        outputs/pvdaq_validation/metrics.parquet               (point scores, cross-check)
        studies/PVDAQ_validation/validation.toml               (model list, excluded sites)
        config/site/<site>.toml, outputs/pvdaq_inventory/*.csv, data/validation/PVDAQ/<sid>/<sid>_report.md
Writes  (next to this file)
        val_numbers.tex    every number quoted in main.tex and supplement.tex, as macros (\\v...)
        val_summary.tex    Table: medians of the reference configurations and PVGIS (main)
        val_compare.tex    Table: paired bootstrap comparisons, site counts (main)
        val_delta.tex      Table: change caused by each refinement-path step (main)
        val_selection.tex  Table: site-selection flow (supplement)
        val_sites.tex      Table: assessment sites (supplement)
        val_scores.tex     Tables: per-site scores with intervals (supplement)
        val_path.tex       Table: cumulative refinement-path medians (supplement)
        outputs/pvdaq_validation/bootstrap.parquet  per-site intervals and paired differences

Bootstrap: paired circular block bootstrap (Kunsch 1989; Politis and Romano 1992) over the
sequence of scored local days of each site, blocks of BLOCK days, REPS replicates, fixed SEED.
Every series of a site is resampled on the same days, so a paired difference carries its own
interval. Intervals are the 2.5th and 97.5th percentiles.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:                     # Python < 3.11
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "pvdaq_validation"
STUDY = ROOT / "studies" / "PVDAQ_validation" / "validation.toml"
SITE_DIR = ROOT / "config" / "site"
INV = ROOT / "outputs" / "pvdaq_inventory"
PVDAQ = ROOT / "data" / "validation" / "PVDAQ"

REPS, BLOCK, SEED = 2000, 5, 20261109
PVGIS = "pvgis|PVGIS-ERA5"
INPUTS = {"point": "ERA5", "nsrdb": "NSRDB", "midc": "Station"}
TAG = {"point": "Era", "nsrdb": "Nsrdb", "midc": "Stn", "pvgis": "Pvgis"}
CFG = {"homer": "Rat", "thesis": "Mod", "current": "Sys", "current:inverter=sandia": "Sdm"}
# table labels: homer = the simple rating model, current = RePV (models.repv), thesis = the
# earlier published-default chain from which the component ablations start
CFG_TEX = {"homer": r"Rating", "thesis": r"Reference chain", "current": r"\RePV",
           "current:inverter=sandia": r"\RePV{} (Sandia)"}

# Evidence class B: the module parameters are not those of the recorded module (surrogate or
# assumed variant). Every other site is class A. Source: the [module] notes of each site file.
CLASS_B = {
    4: "CEC HIP-200BA20 for the recorded HIP-200BA3 (no BA3 entry)",
    33: "60-cell 240~W module for the recorded Silicor 240 (no public datasheet)",
    1199: "STP280-24/Vd assumed for the recorded `Suntech 280W'",
    1200: "mono variant assumed for the recorded `sw240'",
    1202: "mono variant assumed for the recorded `sw240'",
    2107: "STC curve of the HiS-M310KI for the recorded HiS-M310TI (De Soto fit of the TI entry has no positive shunt resistance)",
}
STATE = {"Golden": "CO", "LasVegas": "NV", "Henderson": "NV", "Albuquerque": "NM", "Arbuckle": "CA",
         "CherryHill": "NJ", "Chester": "PA", "Cocoa": "FL", "Gaithersburg": "MD", "HuntValley": "MD",
         "Linthicum": "MD", "PresqueIsle": "ME", "Williston": "VT", "Wilmington": "DE"}
PLACE = {"LasVegas": "Las Vegas", "CherryHill": "Cherry Hill", "HuntValley": "Hunt Valley",
         "PresqueIsle": "Presque Isle", "NewSmyrnaBeach": "New Smyrna Beach", "NewOrleans": "New Orleans",
         "SimonSolar": "Social Circle"}
# sites read from the PVDAQ Data Prize folder rather than the OEDI system list (screen.csv)
PRIZE = {2107: "Arbuckle_2107", 9069: "SimonSolar_9069"}
# configured systems left out after extraction: basis of the decision (metadata, measured
# series only, or measured series against a model run) and reason
EXCLUDED_BASIS = {
    "NewSmyrnaBeach_1231": "metadata",
    "Golden_1332": "metadata",
    "Golden_1284": "measured series",
    "SimonSolar_9069": "measured series",
    "NewOrleans_1275": "measured series and model output",
}
EXCLUDED_TEXT = {
    "NewSmyrnaBeach_1231": "battery-based inverter, so the AC series follows the battery and the load, not the array",
    "Golden_1332": "the documented 1.15~MWp is split between a garage roof and a south facade in an undocumented ratio, and the chain represents one plane",
    "Golden_1284": "hourly peak 1.10 to 1.15 times the DC rating from the module records, with no source for the installed rating",
    "SimonSolar_9069": "the measured series sums an undocumented subset of the plant's AC meters",
    "NewOrleans_1275": "diurnal profile inconsistent with the recorded azimuth of 135$^{\\circ}$ and output about 30\\,\\% below the recorded rating throughout the day, consistent with unrecorded shading",
}
NOT_CONFIGURED = {36: "modules with built-in reflectors, which the chain does not represent"}


# ── helpers ─────────────────────────────────────────────────────────────────

def f1(x, sign=False, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if round(x, nd) == 0:
        return f"{0:.{nd}f}"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def f2(x):
    return f1(x, nd=2)


def thousands(n):
    return f"{int(n):,}".replace(",", "{,}")


def lower_model(c):
    """Model specs are case-insensitive ('site_pressure=True' in outputs, 'true' in the study)."""
    m, sep, spec = c.partition("|")
    return m.lower() + sep + spec if sep else c


def site_label(name):
    place, sid = name.rsplit("_", 1)
    return f"{PLACE.get(place, place)} {sid}"


def load_toml(p):
    with open(p, "rb") as fh:
        return tomllib.load(fh)


class Macros:
    def __init__(self):
        self.lines = []
        self.names = set()

    def __call__(self, name, value, comment=""):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name {name!r} must be letters only")
        if name in self.names:
            raise ValueError(f"macro {name} defined twice")
        self.names.add(name)
        self.lines.append(f"\\newcommand{{\\v{name}}}{{{value}}}" + (f"   % {comment}" if comment else ""))

    def text(self):
        return "\n".join(self.lines) + "\n"


# ── study, sites ────────────────────────────────────────────────────────────

def study():
    st = load_toml(STUDY)
    models = [m.lower() for m in st["models"]]
    steps = {"homer": "RAT"}
    for i, m in enumerate(models[1:12]):           # thesis .. current = P0 .. P10
        steps[m] = f"P{i}"
    steps["current:inverter=sandia"] = "SDM"
    return st, models, steps


def sites_meta(excluded):
    rows = []
    for d in sorted(p for p in (OUT / "sites").glob("*") if (p / "hourly.parquet").exists()):
        name = d.name
        if name in excluded:
            continue
        cfg = load_toml(SITE_DIR / f"{name}.toml")
        sid = int(name.rsplit("_", 1)[1])
        run = json.loads((d / "run.json").read_text())
        loc, arr, sysd = cfg["location"], cfg["array"], cfg["system"]
        snow = cfg.get("snow", {}).get("removal", "nrel")
        rows.append(dict(
            site=name, sid=sid, place=name.rsplit("_", 1)[0], lat=loc["lat"], lon=loc["lon"], alt=loc["altitude"],
            tilt=float(arr["tilt"]), azimuth=float(arr["azimuth"]), dc_kw=float(sysd["rated_capacity_kw"]),
            ac_kw=sysd.get("ac_cap_kw"), year=int(run["window"][0][:4]),
            window=run["window"], station=cfg.get("station", {}).get("midc"),
            n_deg=float(cfg.get("losses", {}).get("n_degradation", 1.0)),
            u_c=cfg.get("thermal", {}).get("pvsyst", {}).get("u_c"),
            fast_snow=(snow == "fast"), tz=cfg["measured"]["tz"],
            cls="B" if sid in CLASS_B else "A",
        ))
    meta = pd.DataFrame(rows).set_index("site")
    meta["group"] = np.where(meta["fast_snow"], "cal", "ind")
    return meta


# ── bootstrap ───────────────────────────────────────────────────────────────

def day_sums(h, tz, series):
    """Per local day: n, So, Soo and per series Ss, Sss, Sso."""
    day = h.index.tz_convert(tz).date
    o = h["measured"].to_numpy(float)
    cols = {"n": np.ones_like(o), "So": o, "Soo": o * o}
    for c in series:
        s = h[c].to_numpy(float)
        cols[c + "#Ss"], cols[c + "#Sss"], cols[c + "#Sso"] = s, s * s, s * o
    return pd.DataFrame(cols, index=h.index).groupby(day).sum()


def scores_from_sums(n, So, Soo, Ss, Sss, Sso):
    bias = 100.0 * (Ss - So) / So
    mse = (Sss - 2 * Sso + Soo) / n
    nrmse = 100.0 * np.sqrt(np.maximum(mse, 0)) / (So / n)
    vs, vo = Sss / n - (Ss / n) ** 2, Soo / n - (So / n) ** 2
    cov = Sso / n - (Ss / n) * (So / n)
    r = cov / np.sqrt(vs * vo)
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (np.sqrt(vs / vo) - 1) ** 2 + (Ss / So - 1) ** 2)
    return bias, nrmse, kge


def block_index(n_days, reps, block, rng):
    nb = math.ceil(n_days / block)
    starts = rng.integers(0, n_days, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_days
    return idx.reshape(reps, nb * block)[:, :n_days]


def bootstrap_site(h, tz, reps, block, rng):
    series = [c for c in h.columns if "|" in c]
    d = day_sums(h, tz, series)
    idx = block_index(len(d), reps, block, rng)
    take = lambda col: d[col].to_numpy()[idx].sum(axis=1)
    full = lambda col: float(d[col].sum())
    n, So, Soo = take("n"), take("So"), take("Soo")
    point, reps_ = {}, {}
    for c in series:
        reps_[c] = scores_from_sums(n, So, Soo, take(c + "#Ss"), take(c + "#Sss"), take(c + "#Sso"))
        point[c] = scores_from_sums(full("n"), full("So"), full("Soo"), full(c + "#Ss"), full(c + "#Sss"), full(c + "#Sso"))
    return point, reps_, len(d)


def pairs(series):
    """(A, B, label): positive differences favour B."""
    out = []
    for spec in INPUTS:
        for a, b in (("homer", "thesis"), ("thesis", "current"), ("homer", "current")):
            A, B = f"{a}|{spec}", f"{b}|{spec}"
            if A in series and B in series:
                out.append((A, B, f"{CFG[a]}{CFG[b]}{TAG[spec]}"))
    for A, B, lab in (("current|point", "current|nsrdb", "EraNsrdb"),
                      ("current|nsrdb", "current|midc", "NsrdbStn"),
                      ("current|point", "current|midc", "EraStn"),
                      ("current|point", PVGIS, "EraPvgis"),
                      (PVGIS, "current|nsrdb", "PvgisNsrdb"),
                      ("current|nsrdb", "current:inverter=sandia|nsrdb", "PvwSdmNsrdb")):
        if A in series and B in series:
            out.append((A, B, lab))
    return out


def run_bootstrap(meta, reps, block):
    rng = np.random.default_rng(SEED)
    rows, prow = [], []
    for name in meta.index:
        h = pd.read_parquet(OUT / "sites" / name / "hourly.parquet")
        h.columns = [lower_model(c) for c in h.columns]
        point, rp, nd = bootstrap_site(h, meta.at[name, "tz"], reps, block, rng)
        for c, (b, e, k) in point.items():
            rb, re_, rk = rp[c]
            model, _, spec = c.partition("|")
            rows.append(dict(site=name, series=c, model=model, input=spec, n_days=nd,
                             bias=b, bias_lo=np.percentile(rb, 2.5), bias_hi=np.percentile(rb, 97.5),
                             nrmse=e, nrmse_lo=np.percentile(re_, 2.5), nrmse_hi=np.percentile(re_, 97.5),
                             kge=k, kge_lo=np.nanpercentile(rk, 2.5), kge_hi=np.nanpercentile(rk, 97.5)))
        for A, B, lab in pairs(list(point)):
            (ba, ea, ka), (bb, eb, kb) = rp[A], rp[B]
            (pba, pea, pka), (pbb, peb, pkb) = point[A], point[B]
            for metric, dr, dp in (("absbias", np.abs(ba) - np.abs(bb), abs(pba) - abs(pbb)),
                                   ("nrmse", ea - eb, pea - peb), ("kge", kb - ka, pkb - pka)):
                lo, hi = np.nanpercentile(dr, 2.5), np.nanpercentile(dr, 97.5)
                prow.append(dict(site=name, pair=lab, A=A, B=B, metric=metric, delta=dp, lo=lo, hi=hi,
                                 verdict="B" if lo > 0 else "A" if hi < 0 else "0"))
        print(f"  {name:<22} {nd} days, {len(point)} series", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(prow)


# ── tables ──────────────────────────────────────────────────────────────────

def med(x):
    x = pd.Series(x).dropna()
    return float(x.median()) if len(x) else float("nan")


def write(name, text):
    (HERE / name).write_text("% generated by make_validation_tables.py; do not edit\n" + text)


def selection(meta, excluded, M):
    scr = pd.read_csv(INV / "screen.csv")
    st = scr["status"].fillna("")
    groups = [
        ("no AC power series in the record", st.str.startswith("excluded: no AC power column")),
        ("net-export meter only", st.str.startswith("excluded: net meter")),
        ("single-axis tracker", st.str.contains("tracker")),
        ("tilt or azimuth not recorded", st.str.startswith("tier G")),
        ("two orientations on one AC series", st.str.startswith("tier M")),
        ("diagnostic clock offset at the $\\pm4$~h search bound", st.str.contains("lag at search bound")),
    ]
    n0 = len(scr) + len(PRIZE)
    taken = sum(int(m.sum()) for _, m in groups)
    left = n0 - taken
    rows = [("PVDAQ systems screened (OEDI system list and Data Prize folder)", "", n0)]
    for lab, m in groups:
        rows.append((f"\\quad excluded: {lab}", "metadata" if "offset" not in lab else "measured series", -int(m.sum())))
    rows.append(("Candidates: fixed single-plane array, AC series, module recorded", "", left))
    for sid, why in NOT_CONFIGURED.items():
        rows.append((f"\\quad not configured: PVDAQ {sid}, {why}", "metadata", -1))
        left -= 1
    for name in excluded:
        rows.append((f"\\quad excluded: {site_label(name)}, {EXCLUDED_TEXT[name]}", EXCLUDED_BASIS[name], -1))
        left -= 1
    rows.append(("Assessment sites", "", left))
    n_cal = int(meta["fast_snow"].sum())
    rows.append(("\\quad of which calibrated (snow removal selected on the scored year)", "measured series and model output", n_cal))
    rows.append(("\\quad of which independent validation", "", left - n_cal))
    M("NScreened", n0, "PVDAQ systems screened")
    M("NCandidates", n0 - taken, "candidates after the metadata screen")
    M("NExcludedLate", len(excluded) + len(NOT_CONFIGURED), "candidates not configured or excluded after extraction")
    M("NConfigured", left, "assessment sites (configured, not excluded)")
    body = "\n".join(f"{a} & {b} & $-${-c} \\\\" if c < 0 else f"{a} & {b} & {c} \\\\" for a, b, c in rows)
    write("val_selection.tex", rf"""\begin{{table}}[htbp]
\centering
\caption{{Site-selection flow. The basis column states what each decision used: metadata only, the measured series only, or the measured series compared with model output. The measured-series exclusions rest on checks that use no model output, except where stated.}}
\label{{tab:s_selection}}
\footnotesize
\begin{{tabularx}}{{\textwidth}}{{>{{\raggedright\arraybackslash}}X >{{\raggedright\arraybackslash}}p{{0.22\textwidth}} r}}
\toprule
Step & Basis & Systems \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{table}}
""")
    return left


def report_shift(sid, year):
    """Clock shift applied in the scored year (standard time / DST), from the extraction report."""
    rep = PVDAQ / str(sid) / f"{sid}_report.md"
    if not rep.exists():
        return "n/a", ""
    txt = rep.read_text()
    m = re.search(r"timestamp shift corrected[^|]*\|[^|]*\|([^|]*)\|", txt)
    out = {}
    if m:
        for part in m.group(1).split(";"):
            part = part.strip()
            status = "S" if part.startswith("standard") else "D"
            for a, b, L in re.findall(r"(\d{4}-\d{2})\.\.(\d{4}-\d{2}) ([+-]\d+) h", part):
                if int(a[:4]) <= year <= int(b[:4]):
                    out.setdefault(status, set()).add(int(L))
    fmt = lambda s: "/".join(f"{x:+d}" if x else "0" for x in sorted(out.get(s, {0})))
    dropped = re.search(r"cannot be aligned, set missing \|[^|]*\|([^|]*)\|", txt)
    nmiss = 0
    if dropped:
        nmiss = sum(1 for x in dropped.group(1).split(",") if x.strip().startswith(str(year)))
    return f"{fmt('S')}; {fmt('D')}".replace("-", "$-$"), nmiss


def sites_table(meta, metrics):
    scr = pd.read_csv(INV / "screen.csv").set_index("system_id")
    summ = pd.read_csv(INV / "summary.csv").set_index("system_id")
    lines = []
    for name, r in meta.sort_values(["place", "sid"]).iterrows():
        mod = (scr["module"].get(r.sid) if r.sid in scr.index else summ["module"].get(r.sid, "n/a"))
        mod = re.sub(r";.*", "", str(mod)).replace("_", " ")
        mod = re.sub(r"\(([^,]*), n=(\d+)\)", lambda m_: f"({m_.group(1)}, {m_.group(2)})", mod)
        shift, nmiss = report_shift(r.sid, r.year)
        hrs = metrics[(metrics.site == name)]["n_hours"].iloc[0]
        cls = r.cls + ("$^{\\mathrm{c}}$" if r.fast_snow else "")
        lines.append(f"{site_label(name)} ({STATE.get(r.place, '')}) & {r.alt:.0f} & "
                     f"{r.tilt:g}/{r.azimuth:g} & {r.dc_kw:,.1f} & {mod} & {cls} & {r.year} & {thousands(hrs)} & "
                     f"{r.station if isinstance(r.station, str) and r.station else 'n/a'} & {r.n_deg:.3f} & {shift} \\\\".replace("_", " "))
    write("val_sites.tex", r"""\begin{table}[htbp]
\centering
\caption{Assessment sites. Tilt and azimuth (degrees from north), DC rating and the recorded module are from the PVDAQ metadata (DC rating: module count times the wattage in the module model name, the recorded system power where the module records lack a wattage, or the documented rating where one is published); class A uses the database entry of the recorded module, class B a surrogate or an assumed variant (Table~\ref{tab:s_classb}); c: calibrated (snow removal selected on the scored year). Hours: scored hours. Station: MIDC station used as the station irradiance input. $f_{\mathrm{deg}}$: age degradation factor. Clock: whole-hour shift applied by the extraction in the scored year, standard time; daylight-saving time.}
\label{tab:s_sites}
\scriptsize
\setlength{\tabcolsep}{1.5pt}
\makebox[\textwidth][c]{\begin{tabular}{l r l r >{\raggedright\arraybackslash}p{0.23\textwidth} l r r l r l}
\toprule
Site & Alt. (m) & Tilt/az. & DC (kWp) & Recorded module (technology, count) & Cl. & Year & Hours & Station & $f_{\mathrm{deg}}$ & Clock (h) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[htbp]
\centering
\caption{Class-B sites: surrogate module or assumed variant.}
\label{tab:s_classb}
\footnotesize
\begin{tabularx}{\textwidth}{l X}
\toprule
Site & Module parameters used \\
\midrule
""" + "\n".join(f"{site_label(n)} & {CLASS_B[r.sid]} \\\\" for n, r in meta.iterrows() if r.sid in CLASS_B) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")


def summary_tables(meta, boot, pairs_df, metrics, steps, M):
    ind = meta.index[meta.group == "ind"]
    cal = meta.index[meta.group == "cal"]
    stn = [s for s in ind if ((boot.site == s) & (boot.series == "current|midc")).any()]
    stn_all = [s for s in meta.index if ((boot.site == s) & (boot.series == "current|midc")).any()]
    b = boot.set_index(["site", "series"])

    def vals(series, sites, col):
        return pd.Series([b.at[(s, series), col] for s in sites if (s, series) in b.index], dtype=float)

    # counts
    M("NSites", len(meta), "assessment sites with outputs")
    M("NInd", len(ind), "independent validation sites")
    M("NCal", len(cal), "calibrated sites")
    M("NClassA", int(((meta.cls == "A") & (meta.group == "ind")).sum()))
    M("NClassB", int(((meta.cls == "B") & (meta.group == "ind")).sum()))
    M("NStation", len(stn), "independent sites with station irradiance")
    M("NStationAll", len(stn_all))
    M("NStationGolden", sum(1 for s in stn_all if s.startswith("Golden")))
    M("NStationLV", sum(1 for s in stn_all if not s.startswith("Golden")))
    M("NHours", thousands(metrics.groupby("site")["n_hours"].first().loc[meta.index].sum()), "scored hours, all sites")
    M("HoursMin", thousands(metrics.groupby("site")["n_hours"].first().loc[meta.index].min()))
    M("HoursMax", thousands(metrics.groupby("site")["n_hours"].first().loc[meta.index].max()))
    M("DcMin", f"{meta.dc_kw.min():.1f}")
    M("DcMax", f"{meta.dc_kw.max():.0f}")
    M("YearMin", meta.year.min())
    M("YearMax", meta.year.max())
    M("AltMax", f"{meta.alt.max():,.0f}".replace(",", "{,}"))
    M("DegMin", f"{meta.n_deg.min():.3f}")
    M("DegMax", f"{meta.n_deg.max():.3f}")
    M("NSdm", int((boot.model == "current:inverter=sandia").groupby(boot.site).any().sum()))
    M("Reps", thousands(REPS_USED))
    M("Block", BLOCK)
    M("Seed", SEED)

    # main summary: independent sites
    rows = []
    for model in ("homer", "thesis", "current"):
        for spec in ("point", "nsrdb"):
            c = f"{model}|{spec}"
            bias, nr, kg = vals(c, ind, "bias"), vals(c, ind, "nrmse"), vals(c, ind, "kge")
            key = CFG[model] + TAG[spec]
            M(f"Med{key}Bias", f1(med(bias), sign=True)); M(f"Med{key}AbsBias", f1(med(bias.abs())))
            M(f"Med{key}Nrmse", f1(med(nr))); M(f"Med{key}Kge", f2(med(kg)))
            M(f"Five{key}", int((bias.abs() <= 5).sum())); M(f"Ten{key}", int((bias.abs() <= 10).sum()))
            rows.append((CFG_TEX[model], INPUTS[spec], bias, nr, kg))
    bias, nr, kg = vals(PVGIS, ind, "bias"), vals(PVGIS, ind, "nrmse"), vals(PVGIS, ind, "kge")
    M("MedPvgisBias", f1(med(bias), sign=True)); M("MedPvgisAbsBias", f1(med(bias.abs())))
    M("MedPvgisNrmse", f1(med(nr))); M("MedPvgisKge", f2(med(kg)))
    M("FivePvgis", int((bias.abs() <= 5).sum())); M("TenPvgis", int((bias.abs() <= 10).sum()))
    M("NPvgis", len(bias))
    rows.append(("PVGIS", "PVGIS-ERA5", bias, nr, kg))

    def iqr(x):
        return f"{f1(x.quantile(0.25), sign=True)} to {f1(x.quantile(0.75), sign=True)}"

    body = []
    for i, (cfg, inp, bias, nr, kg) in enumerate(rows):
        if i and rows[i - 1][0] != cfg:
            body.append(r"\addlinespace[2pt]")
        body.append(f"{cfg} & {inp} & {f1(med(bias), sign=True)} & {f1(med(bias.abs()))} & "
                    f"{f1(med(nr))} & {f2(med(kg))} \\\\")
    write("val_summary.tex", r"""\begin{table}[htbp]
\centering
\caption{Median scores against measured AC output across the \vNInd{} independent validation sites. PVGIS-ERA5 is scored at the \vNPvgis{} sites whose scored year lies in its database period. Interquartile ranges and site counts are given in the repository supplement.}
\label{tab:val_summary}
\small
\begin{tabular}{l l r r r r}
\toprule
Series & Irradiance & $b$ (\%) & $|b|$ (\%) & NRMSE (\%) & KGE \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # calibrated sites, station subset
    for grp, sites in (("Cal", cal), ("Stn", stn)):
        for c, key in (("current|point", "SysEra"), ("current|nsrdb", "SysNsrdb"), ("current|midc", "SysStn"),
                       (PVGIS, "Pvgis"), ("homer|nsrdb", "RatNsrdb"), ("thesis|nsrdb", "ModNsrdb")):
            bias, nr, kg = vals(c, sites, "bias"), vals(c, sites, "nrmse"), vals(c, sites, "kge")
            if len(bias) == 0:
                continue
            M(f"{grp}{key}Bias", f1(med(bias), sign=True)); M(f"{grp}{key}AbsBias", f1(med(bias.abs())))
            M(f"{grp}{key}Nrmse", f1(med(nr))); M(f"{grp}{key}Kge", f2(med(kg)))

    # paired comparisons
    P = pairs_df[pairs_df.site.isin(ind)]
    labels = [
        ("RatModEra", r"Rating $\to$ Ref.", "ERA5"), ("ModSysEra", r"Ref. $\to$ \RePV{}", "ERA5"),
        ("RatModNsrdb", r"Rating $\to$ Ref.", "NSRDB"), ("ModSysNsrdb", r"Ref. $\to$ \RePV{}", "NSRDB"),
        ("RatModStn", r"Rating $\to$ Ref.", "station"), ("ModSysStn", r"Ref. $\to$ \RePV{}", "station"),
        ("EraNsrdb", r"ERA5 $\to$ NSRDB", r"\RePV{}"), ("NsrdbStn", r"NSRDB $\to$ station", r"\RePV{}"),
        ("EraPvgis", r"\RePV{} ERA5 $\to$ PVGIS-ERA5", "ERA5"), ("PvgisNsrdb", r"PVGIS-ERA5 $\to$ \RePV{} NSRDB", "both"),
    ]
    body = []
    for lab, a, inp in labels:
        q = P[P.pair == lab]
        if q.empty:
            continue
        n = q.site.nunique()
        cells = []
        for metric in ("absbias", "nrmse", "kge"):
            v = q[q.metric == metric].verdict.value_counts()
            nb, na, n0 = int(v.get("B", 0)), int(v.get("A", 0)), int(v.get("0", 0))
            cells.append(f"{nb} & {na} & {n0}")
            mk = {"absbias": "Abs", "nrmse": "Nrmse", "kge": "Kge"}[metric]
            M(f"Pair{lab}{mk}Better", nb); M(f"Pair{lab}{mk}Worse", na); M(f"Pair{lab}{mk}Tie", n0)
            M(f"Pair{lab}{mk}Med", f1(med(q[q.metric == metric].delta), sign=True, nd=2 if metric == "kge" else 1))
        M(f"Pair{lab}N", n)
        body.append(f"{a} & {inp} & {n} & " + " & ".join(cells) + r" \\")
    write("val_compare.tex", r"""\begin{table}[htbp]
\centering
\caption{Paired comparisons at the independent validation sites. For each change from series $A$ to series $B$, the number of sites at which the 95\,\% paired-bootstrap interval of the difference (Eq.~\ref{eq:paired}) lay above zero ($B$ closer to measured output), below zero ($A$ closer), or included zero (U, unresolved). $n$: sites at which both series were scored.}
\label{tab:val_compare}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l l r ccc ccc ccc}
\toprule
 & & & \multicolumn{3}{c}{$|b|$} & \multicolumn{3}{c}{NRMSE} & \multicolumn{3}{c}{KGE} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}
Change $A\to B$ & Irradiance & $n$ & $B$ & $A$ & U & $B$ & $A$ & U & $B$ & $A$ & U \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # per-step deltas (independent sites), cumulative medians
    inv = {v: k for k, v in steps.items()}
    order = ["RAT"] + [f"P{i}" for i in range(11)]
    body, cum = [], []
    STEP_TEX = {"RAT": r"Rating", "P0": r"$\mathsf{P}_0$ (Ref.)", "P10": r"$\mathsf{P}_{10}$ (\RePV{})", "SDM": r"\RePV{} (Sandia)"}
    for i, st in enumerate(order + ["SDM"]):
        model = inv[st]
        tex = STEP_TEX.get(st, f"$\\mathsf{{P}}_{{{st[1:]}}}$")
        cumcells = []
        for spec in ("point", "nsrdb", "midc"):
            sites = stn if spec == "midc" else ind
            bias = vals(f"{model}|{spec}", sites, "bias")
            nr = vals(f"{model}|{spec}", sites, "nrmse")
            cumcells += [f1(med(bias.abs())), f1(med(nr))]
        if st != "SDM":
            cum.append(f"{tex} & " + " & ".join(cumcells) + r" \\")
        if st in ("RAT",):
            continue
        prev = inv[order[i - 1]] if st != "SDM" else inv["P10"]
        cells = []
        for spec in ("point", "nsrdb"):
            dl, dn, act = [], [], 0
            for s in ind:
                ka, kb = (s, f"{prev}|{spec}"), (s, f"{model}|{spec}")
                if ka in b.index and kb in b.index:
                    db = b.at[kb, "bias"] - b.at[ka, "bias"]
                    dl.append(db); dn.append(b.at[kb, "nrmse"] - b.at[ka, "nrmse"])
                    act += abs(db) >= 0.1
            dl, dn = pd.Series(dl, dtype=float), pd.Series(dn, dtype=float)
            rng_ = f"{f1(dl.min(), sign=True)} to {f1(dl.max(), sign=True)}" if len(dl) else "n/a"
            cells.append(f"{f1(med(dl), sign=True, nd=2)} & {rng_} & {f1(med(dn), sign=True, nd=2)} & {act}/{len(dl)}")
            key = st.replace("P", "Step") if st != "SDM" else "StepSdm"
            key = key.replace("10", "Ten").replace("0", "Zero").replace("1", "One").replace("2", "Two").replace("3", "Three") \
                     .replace("4", "Four").replace("5", "Five").replace("6", "Six").replace("7", "Seven").replace("8", "Eight").replace("9", "Nine")
            M(f"{key}{TAG[spec]}Bias", f1(med(dl), sign=True, nd=2)); M(f"{key}{TAG[spec]}Nrmse", f1(med(dn), sign=True, nd=2))
            M(f"{key}{TAG[spec]}Min", f1(dl.min(), sign=True)); M(f"{key}{TAG[spec]}Max", f1(dl.max(), sign=True))
            M(f"{key}{TAG[spec]}Act", act)
        body.append(f"{tex} & " + " & ".join(cells) + r" \\")
    write("val_delta.tex", r"""\begin{table}[htbp]
\centering
\caption{Change caused by each step at the \vNInd{} independent validation sites: median and range of the change in signed bias $\Delta b$ (percentage points), median change in NRMSE $\Delta$NRMSE (points), and the number of sites at which the step changed the bias by at least 0.1 points. $\mathsf{P}_0$ is measured from Rating, each later step from the step before it, and \RePV{} (Sandia) from $\mathsf{P}_{10}$ at the sites with a CEC inverter entry.}
\label{tab:ladder_delta}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l rcrr rcrr}
\toprule
 & \multicolumn{4}{c}{ERA5} & \multicolumn{4}{c}{NSRDB} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Step & $\Delta b$ & range & $\Delta$NRMSE & sites & $\Delta b$ & range & $\Delta$NRMSE & sites \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("val_path.tex", r"""\begin{table}[htbp]
\centering
\caption{Cumulative refinement path: median absolute bias $|b|$ and median NRMSE (\%) of every step, on ERA5 and NSRDB irradiance at the \vNInd{} independent validation sites and on station irradiance at the \vNStation{} independent sites with an MIDC station.}
\label{tab:ladder}
\small
\begin{tabular}{l rr rr rr}
\toprule
 & \multicolumn{2}{c}{ERA5} & \multicolumn{2}{c}{NSRDB} & \multicolumn{2}{c}{Station} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Step & $|b|$ & NRMSE & $|b|$ & NRMSE & $|b|$ & NRMSE \\
\midrule
""" + "\n".join(cum) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # per-site scores with intervals (supplement)
    def cell(site, series, col, nd=1):
        k = (site, series)
        if k not in b.index:
            return "n/a"
        r = b.loc[k]
        return f"{f1(r[col], sign=(col == 'bias'), nd=nd)} [{f1(r[col + '_lo'], sign=(col == 'bias'), nd=nd)}, {f1(r[col + '_hi'], sign=(col == 'bias'), nd=nd)}]"

    def pt(site, series, col, nd=1):
        k = (site, series)
        return f1(b.at[k, col], sign=(col == "bias"), nd=nd) if k in b.index else "n/a"

    tabs = []
    for col, title, lab in (("bias", "Signed bias $b$ (\\%)", "tab:s_scores"), ("nrmse", "NRMSE (\\%)", "tab:s_scores_nrmse")):
        lines = []
        for grp, sites in (("Independent validation sites", ind), ("Calibrated sites", cal)):
            lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{grp}}}}} \\\\")
            for s in sorted(sites, key=lambda x: (meta.at[x, "place"], meta.at[x, "sid"])):
                lines.append(f"{site_label(s)} & {pt(s, 'homer|point', col)} & "
                             f"{cell(s, 'current|point', col)} & {cell(s, 'current|nsrdb', col)} & {cell(s, 'current|midc', col)} & "
                             f"{cell(s, PVGIS, col)} & {pt(s, 'current:inverter=sandia|nsrdb', col)} \\\\")
        tabs.append(rf"""\begin{{table}}[htbp]
\centering
\caption{{{title} per site, with 95\,\% block-bootstrap intervals for \RePV{{}} on each irradiance input and for PVGIS-ERA5; the simple rating model on ERA5 and \RePV{{}} with the Sandia inverter model on NSRDB (sites with a CEC inverter entry) without intervals. Paired differences are in \texttt{{outputs/pvdaq\_validation/bootstrap.parquet}} of the archived release.}}
\label{{{lab}}}
\scriptsize
\setlength{{\tabcolsep}}{{1.4pt}}
\makebox[\textwidth][c]{{\begin{{tabular}}{{l r llll r}}
\toprule
Site & Rating ERA5 & \RePV{{}} ERA5 & \RePV{{}} NSRDB & \RePV{{}} station & PVGIS-ERA5 & Sandia \\
\midrule
{chr(10).join(l.replace('multicolumn{9}', 'multicolumn{7}') for l in lines)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")
    write("val_scores.tex", "\n".join(tabs))

    # sites within tolerance, extremes (independent, SYS-PVW NSRDB)
    bias = pd.Series({s: b.at[(s, "current|nsrdb"), "bias"] for s in ind})
    M("SysNsrdbBiasLo", f1(bias.min(), sign=True)); M("SysNsrdbBiasHi", f1(bias.max(), sign=True))
    # refinement against irradiance input: NRMSE change of the path (RAT -> SYS, MOD -> SYS, ERA5)
    # and of the input change ERA5 -> NSRDB (SYS-PVW), per site
    gr, gp, gi = [], [], []
    for s in ind:
        ks = [(s, "homer|point"), (s, "thesis|point"), (s, "current|point"), (s, "current|nsrdb")]
        if all(k in b.index for k in ks):
            r_, m_, e_, n_ = (b.at[k, "nrmse"] for k in ks)
            gr.append(r_ - e_); gp.append(m_ - e_); gi.append(e_ - n_)
    M("PathRatNrmseGainEra", f1(med(gr))); M("PathModNrmseGainEra", f1(med(gp))); M("InputNrmseGain", f1(med(gi)))
    M("InputNrmseGainLo", f1(min(gi))); M("InputNrmseGainHi", f1(max(gi)))
    M("NInputBeatsPath", sum(i > r_ for r_, i in zip(gr, gi)))
    # regional groups (independent sites), SYS-PVW bias by input
    groups = {"Golden": [s for s in ind if meta.at[s, "place"] == "Golden"],
              "Desert": [s for s in ind if meta.at[s, "place"] in ("LasVegas", "Henderson")],
              "East": [s for s in ind if meta.at[s, "lon"] > -90]}
    for g, sites in groups.items():
        M(f"N{g}", len(sites))
        for spec in ("point", "nsrdb", "midc"):
            v = pd.Series([b.at[(s, f"current|{spec}"), "bias"] for s in sites if (s, f"current|{spec}") in b.index], dtype=float)
            if len(v):
                M(f"{g}{TAG[spec]}BiasLo", f1(v.min(), sign=True)); M(f"{g}{TAG[spec]}BiasHi", f1(v.max(), sign=True))
                M(f"{g}{TAG[spec]}BiasMed", f1(med(v), sign=True))
        v = pd.Series([b.at[(s, PVGIS), "bias"] for s in sites if (s, PVGIS) in b.index], dtype=float)
        if len(v):
            M(f"{g}PvgisBiasLo", f1(v.min(), sign=True)); M(f"{g}PvgisBiasHi", f1(v.max(), sign=True))
    # terrain horizon of the sites (PVGIS profiles in the cache)
    hz = []
    for s in meta.index:
        fp = ROOT / "cache" / "horizon" / f"pvgis_horizon__{meta.at[s, 'lat']}__{meta.at[s, 'lon']}.parquet"
        if fp.exists():
            hz.append(float(pd.read_parquet(fp).iloc[:, 0].max()))
    if hz:
        M("HorizonMax", f1(max(hz))); M("NHorizon", len(hz))
    # winners per pair on point values (no interval)
    for lab, A, B in (("PvgisBeatsSysEra", "current|point", PVGIS), ("NsrdbBeatsPvgis", PVGIS, "current|nsrdb"),
                      ("NsrdbBeatsEra", "current|point", "current|nsrdb")):
        n = sum(1 for s in ind if (s, A) in b.index and (s, B) in b.index and b.at[(s, B), "nrmse"] < b.at[(s, A), "nrmse"])
        M(lab + "Nrmse", n)


FEATURES = [   # (label, macro key, MOD with the feature, MOD without, SYS with, SYS without)
    ("Site-pressure air mass", "Pres", "thesis:site_pressure=true", "thesis", "current", "current:site_pressure=false"),
    ("Diffuse IAM", "Iam", "thesis:iam=physical", "thesis", "current", "current:iam=physical_beam"),
    ("Terrain horizon", "Hor", "thesis:use_horizon=true", "thesis", "current", "current:use_horizon=false"),
    ("Physical loss placement", "Loss", "thesis:loss_order=physical", "thesis", "current", "current:loss_order=post_inverter"),
    ("Optical soiling", "Soil", "thesis:soiling_position=optical", "thesis", "current", "current:soiling_position=post_ac"),
    ("AC capacity limit", "Clip", "thesis:ac_clip=true", "thesis", "current", "current:ac_clip=false"),
    ("Snow-depth condition and reset", "Depth", "thesis:snow_use_depth=true", "thesis", "current", "current:snow_use_depth=false"),
    ("6~h snowfall trigger", "Trig", "thesis:snow_trigger=rolling", "thesis", "current", "current:snow_trigger=hourly"),
    ("Snow-dependent albedo", "Alb", "thesis:snow_albedo=0.6", "thesis", "current", "current:snow_albedo=none"),
    ("NREL snow model", "Snow", "thesis", "thesis:snow=none", "current", "current:snow=none"),
]


def ablation_table(meta, boot, M):
    ind = meta.index[meta.group == "ind"]
    b = boot.set_index(["site", "series"])
    body = []
    for label, key, mw, mo, sw, so in FEATURES:
        cells = []
        for spec in ("point", "nsrdb"):
            for direction, (w, o) in (("Mod", (mw, mo)), ("Sys", (sw, so))):
                d = []
                for s in ind:
                    kw_, ko = (s, f"{w}|{spec}"), (s, f"{o}|{spec}")
                    if kw_ in b.index and ko in b.index:
                        d.append(b.at[kw_, "bias"] - b.at[ko, "bias"])
                d = pd.Series(d, dtype=float)
                act = int((d.abs() >= 0.1).sum())
                M(f"Abl{key}{direction}{TAG[spec]}", f1(med(d), sign=True, nd=2))
                M(f"Abl{key}{direction}{TAG[spec]}Lo", f1(d.min(), sign=True)); M(f"Abl{key}{direction}{TAG[spec]}Hi", f1(d.max(), sign=True))
                M(f"Abl{key}{direction}{TAG[spec]}Act", act)
                if spec == "nsrdb":
                    cells.append(f"{f1(med(d), sign=True, nd=2)} & {f1(d.min(), sign=True)} to {f1(d.max(), sign=True)} & {act}/{len(d)}")
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    write("val_ablation.tex", r"""\begin{table}[htbp]
\centering
\caption{One-at-a-time ablations at the \vNInd{} independent validation sites on NSRDB irradiance: change in signed bias $\Delta b$ (percentage points) caused by each feature when it is added to Ref. alone and when it is removed from \RePV{} alone (median and range across sites), and the number of sites at which it changed the bias by at least 0.1 points. A positive $\Delta b$ means the feature raised the simulated energy.}
\label{tab:s_ablation_ref}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l rcr rcr}
\toprule
 & \multicolumn{3}{c}{Added to Ref.} & \multicolumn{3}{c}{Removed from \RePV{}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Feature & $\Delta b$ & range & sites & $\Delta b$ & range & sites \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# MIDC station coordinates, from the station pages (midcdmz.nrel.gov, BMS and UNLV)
STATIONS = {"BMS": (39.742, -105.18), "UNLV": (36.107, -115.1425)}


def km(a, b):
    (la1, lo1), (la2, lo2) = np.radians(a), np.radians(b)
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(h)))


def provenance_macros(meta, M):
    """Station distances and the source of U_c, from the site files."""
    for st, xy in STATIONS.items():
        d = [km((meta.at[s, "lat"], meta.at[s, "lon"]), xy) for s in meta.index if meta.at[s, "station"] == st]
        if d:
            M(f"{st.title()}DistMin", f1(min(d))); M(f"{st.title()}DistMax", f1(max(d), nd=0))
    src = {"fit": 0, "noct": 0, "class": 0}
    for s in meta.index:
        txt = (SITE_DIR / f"{s}.toml").read_text()
        line = next((l for l in txt.splitlines() if l.strip().startswith("u_c")), "")
        src["noct" if "NOCT" in line else "fit" if "fit to" in line else "class"] += 1
    M("UcFit", src["fit"]); M("UcNoct", src["noct"]); M("UcClass", src["class"])


def make_figure(meta, boot):
    """Per-site |b| and NRMSE of RAT, MOD, SYS-PVW on each input and of PVGIS-ERA5."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib missing: figure skipped")
        return
    b = boot.set_index(["site", "series"])
    stn = {s for s in meta.index if (s, "current|midc") in b.index}
    cols = {"homer": "#2a78d6", "thesis": "#eb6834", "current": "#1baf7a", "pvgis": "#eda100"}
    marks = {"homer": "o", "thesis": "s", "current": "D", "pvgis": "^"}
    names = {"homer": "RAT", "thesis": "MOD", "current": "SYS", "pvgis": "PVGIS"}
    slots = [("point", m) for m in ("homer", "thesis", "current")] + [("pvgis", "pvgis")] + \
            [("nsrdb", m) for m in ("homer", "thesis", "current")] + [("midc", m) for m in ("homer", "thesis", "current")]
    xs, x = [], 0.0
    for i, (spec, m) in enumerate(slots):
        grp_of = lambda sp: "point" if sp == "pvgis" else sp
        if i and grp_of(slots[i - 1][0]) != grp_of(spec):
            x += 0.8
        xs.append(x)
        x += 1.0
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
    for ax, col, lab in ((axes[0], "bias", r"$|b|$ (%)"), (axes[1], "nrmse", "NRMSE (%)")):
        for xi, (spec, m) in zip(xs, slots):
            series = PVGIS if m == "pvgis" else f"{m}|{spec}"
            for grp, filled in (("ind", True), ("cal", False)):
                sites = [s for s in meta.index[meta.group == grp] if (s, series) in b.index]
                v = np.array([b.at[(s, series), col] for s in sites], float)
                v = np.abs(v) if col == "bias" else v
                if not len(v):
                    continue
                jx = xi + rng.uniform(-0.22, 0.22, len(v))
                ax.scatter(jx, v, s=22, marker=marks[m], linewidths=1.0, zorder=3,
                           facecolors=cols[m] if filled else "white", edgecolors=cols[m])
                if grp == "ind":
                    ax.plot([xi - 0.32, xi + 0.32], [np.median(v)] * 2, color="#1f1f1e", lw=2, zorder=4,
                            solid_capstyle="round")
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([names[m] for _, m in slots], fontsize=8)
    groups = {"ERA5": (xs[0], xs[3]), "NSRDB": (xs[4], xs[6]), "Station": (xs[7], xs[9])}
    for g, (a, c) in groups.items():
        axes[1].annotate(g, xy=((a + c) / 2, 0), xycoords=("data", "axes fraction"), xytext=(0, -24),
                         textcoords="offset points", ha="center", va="top", fontsize=9)
    n_ind, n_stn = int((meta.group == "ind").sum()), len([s for s in stn if meta.at[s, "group"] == "ind"])
    axes[0].set_title(f"Independent sites (filled; bar = median), calibrated sites (open); "
                      f"station: {n_stn} independent sites", fontsize=8, loc="left", color="#55554f")
    fig.tight_layout()
    fig.savefig(HERE / "fig_validation.pdf")
    plt.close(fig)


def main(argv=None):
    global REPS_USED
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args(argv)
    REPS_USED = a.reps
    st, models, steps = study()
    excluded = dict(st.get("excluded_sites", {}))
    meta = sites_meta(excluded)
    metrics = pd.read_parquet(OUT / "metrics.parquet")
    metrics["model"] = metrics["model"].str.lower()
    print(f"{len(meta)} sites; bootstrap {a.reps} replicates, {BLOCK}-day blocks, seed {SEED}")
    boot, pr = run_bootstrap(meta, a.reps, BLOCK)
    # the bootstrap's point scores must equal the runner's
    chk = boot.merge(metrics.rename(columns={"bias_pct": "b0", "nrmse_pct": "e0", "kge": "k0"})[["site", "model", "input", "b0", "e0", "k0"]],
                     on=["site", "model", "input"], how="left")
    if chk.b0.isna().any():
        sys.exit(f"series without a runner score: {chk[chk.b0.isna()][['site', 'series']].values.tolist()[:5]}")
    worst = float((chk.bias - chk.b0).abs().max())
    if not worst < 1e-6:
        sys.exit(f"bootstrap point scores differ from metrics.parquet by {worst}")
    boot.to_parquet(OUT / "bootstrap.parquet")
    pr.to_parquet(OUT / "bootstrap_pairs.parquet")
    M = Macros()
    selection(meta, excluded, M)
    sites_table(meta, metrics)
    summary_tables(meta, boot, pr, metrics, steps, M)
    ablation_table(meta, boot, M)
    provenance_macros(meta, M)
    make_figure(meta, boot)
    write("val_numbers.tex", "% Every assessment number quoted in main.tex and supplement.tex.\n"
          f"% Sites: {len(meta)} ({', '.join(meta.index)}).\n" + M.text())
    print(f"wrote {len(M.names)} macros and the val_*.tex tables to {HERE}")


if __name__ == "__main__":
    main()
