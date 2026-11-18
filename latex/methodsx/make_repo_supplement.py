"""Write supplementary.md, the repository supplement of the MethodsX article, from val_numbers.tex.

    python make_repo_supplement.py        # after make_validation_tables.py

Every number in supplementary.md comes from the macros that make_validation_tables.py
writes to val_numbers.tex, so the file cannot drift from the assessment outputs.
Never edit supplementary.md by hand; edit the text below.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = dict(re.findall(r"^\\newcommand\{\\v(\w+)\}\{(.*)\}[ \t]*(?:%.*)?$", (HERE / "val_numbers.tex").read_text(), re.M))


def g(key: str) -> str:
    """A macro value as plain text (LaTeX minus signs and thousands separators undone)."""
    return V[key].replace("$-$", "−").replace("{,}", ",")


SERIES = [("Rating", "ERA5", "RatEra"), ("Rating", "NSRDB", "RatNsrdb"), ("RePV", "ERA5", "SysEra"),
          ("RePV", "NSRDB", "SysNsrdb"), ("PVGIS", "PVGIS-ERA5", "Pvgis")]
FEATURES = [("Site-pressure air mass", "Pres"), ("Diffuse IAM", "Iam"), ("Terrain horizon", "Hor"),
            ("Physical loss placement", "Loss"), ("Soiling on irradiance", "Soil"), ("AC capacity limit", "Clip"),
            ("Snow-depth condition and reset", "Depth"), ("6 h snowfall trigger", "Trig"),
            ("Snow-dependent albedo", "Alb"), ("Complete snow model", "Snow")]
PAIRS = [("RePV ERA5 → RePV NSRDB", "EraNsrdb"), ("RePV NSRDB → RePV station", "NsrdbStn"),
         ("RePV ERA5 → PVGIS-ERA5", "EraPvgis"), ("PVGIS-ERA5 → RePV NSRDB", "PvgisNsrdb")]


def table(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(r) + " |" for r in rows])


def main():
    n = g("NInd")
    summary = table(["Series", "Irradiance", "Median b (%)", "Median \\|b\\| (%)", "Sites \\|b\\| ≤ 5 %", "Sites \\|b\\| ≤ 10 %",
                     "Median NRMSE (%)", "Median KGE"],
                    [[s, i, g(f"Med{k}Bias"), g(f"Med{k}AbsBias"), g(f"Five{k}"), g(f"Ten{k}"), g(f"Med{k}Nrmse"),
                      g(f"Med{k}Kge")] for s, i, k in SERIES])
    pairs = table(["Change A → B", "n", "\\|b\\|: B / A / U", "NRMSE: B / A / U", "KGE: B / A / U",
                   "Median Δ\\|b\\|", "Median ΔNRMSE"],
                  [[lab, g(f"Pair{k}N")] + [f"{g(f'Pair{k}{m}Better')} / {g(f'Pair{k}{m}Worse')} / {g(f'Pair{k}{m}Tie')}"
                                            for m in ("Abs", "Nrmse", "Kge")] + [g(f"Pair{k}AbsMed"), g(f"Pair{k}NrmseMed")]
                   for lab, k in PAIRS])
    abl = {}
    for inp, tag in (("ERA5", "Era"), ("NSRDB", "Nsrdb")):
        abl[inp] = table(["Component", "Removed from RePV: median Δb", "Range", "Sites",
                          "Added to the reference chain: median Δb", "Range", "Sites"],
                         [[lab, g(f"Abl{k}Sys{tag}"), f"{g(f'Abl{k}Sys{tag}Lo')} to {g(f'Abl{k}Sys{tag}Hi')}",
                           f"{g(f'Abl{k}Sys{tag}Act')}/{n}", g(f"Abl{k}Mod{tag}"),
                           f"{g(f'Abl{k}Mod{tag}Lo')} to {g(f'Abl{k}Mod{tag}Hi')}", f"{g(f'Abl{k}Mod{tag}Act')}/{n}"]
                          for lab, k in FEATURES])
    regions = table(["Region", "Sites", "ERA5 median (range)", "NSRDB median (range)", "Station median (range)",
                     "PVGIS-ERA5 range"],
                    [["Golden, Colorado Front Range", g("NGolden")] +
                     [f"{g(f'Golden{t}BiasMed')} ({g(f'Golden{t}BiasLo')} to {g(f'Golden{t}BiasHi')})" for t in ("Era", "Nsrdb", "Stn")] +
                     [f"{g('GoldenPvgisBiasLo')} to {g('GoldenPvgisBiasHi')}"],
                     ["Southern Nevada", g("NDesert")] +
                     [f"{g(f'Desert{t}BiasMed')} ({g(f'Desert{t}BiasLo')} to {g(f'Desert{t}BiasHi')})" for t in ("Era", "Nsrdb", "Stn")] +
                     [f"{g('DesertPvgisBiasLo')} to {g('DesertPvgisBiasHi')}"],
                     ["East of the Mississippi", str(int(n) - int(g("NGolden")) - int(g("NDesert")) - 2)] +
                     [f"{g(f'East{t}BiasMed')} ({g(f'East{t}BiasLo')} to {g(f'East{t}BiasHi')})" for t in ("Era", "Nsrdb")] +
                     ["n/a", f"{g('EastPvgisBiasLo')} to {g('EastPvgisBiasHi')}"]])
    station = table(["Series", "Irradiance", "Median b (%)", "Median \\|b\\| (%)", "Median NRMSE (%)", "Median KGE"],
                    [[s, i, g(f"Stn{k}Bias"), g(f"Stn{k}AbsBias"), g(f"Stn{k}Nrmse"), g(f"Stn{k}Kge")]
                     for s, i, k in (("Rating", "NSRDB", "RatNsrdb"), ("RePV", "ERA5", "SysEra"), ("RePV", "NSRDB", "SysNsrdb"),
                                     ("RePV", "station", "SysStn"), ("PVGIS", "PVGIS-ERA5", "Pvgis"))])
    calibrated = table(["Series", "Irradiance", "Median b (%)", "Median \\|b\\| (%)", "Median NRMSE (%)", "Median KGE"],
                       [[s, i, g(f"Cal{k}Bias"), g(f"Cal{k}AbsBias"), g(f"Cal{k}Nrmse"), g(f"Cal{k}Kge")]
                        for s, i, k in (("Rating", "NSRDB", "RatNsrdb"), ("RePV", "ERA5", "SysEra"),
                                        ("RePV", "NSRDB", "SysNsrdb"), ("PVGIS", "PVGIS-ERA5", "Pvgis"))])

    text = f"""# RePV: repository supplement to the MethodsX article

This file holds material that is not part of the article (`main.tex`) or its journal supplement (`supplement.tex`): the verification tests with their acceptance criteria, the reproducibility settings, implementation notes and the extended results. Table and section numbers prefixed S refer to the journal supplement; all other numbers refer to the article. It is written by `latex/methodsx/make_repo_supplement.py` from `val_numbers.tex`; do not edit it by hand.

**Names.** RePV is the model `repv` of the Python package `solarmodel` (`solarmodel.models.repv()`, identical to `current()`). The simple rating model of the validation is `rating` (identical to `homer()`). The reference chain of the ablations is `thesis`: RePV without site-pressure air mass, diffuse IAM, terrain horizon, physical loss placement, soiling on the irradiance, AC capacity limit, ground-snow-depth condition, 6 h trigger and snow albedo.

**Terminology.** *Verification*: tests of the implementation against reference implementations, limiting cases and stored outputs (Section 1). *Independent validation*: the {n} sites configured without reference to measured output. *Calibrated assessment*: the {g('NCal')} Mid-Atlantic sites whose fast snow removal was set after comparison with measured output. *Reference model*: PVGIS-ERA5, an independently developed model, not an observational reference.

---

## 1. Verification tests

`python -m pytest` runs the suite (`tests/`). With the pinned environment of Section 2, every test passes. An internal equivalence test against the author's thesis results is not part of the public release, because its input data belong to a third party.

| Test file | Quantity tested | Acceptance criterion |
|---|---|---|
| `test_chain.py` | Reindl, Perez, Hay-Davies and isotropic transposition with a flat horizon, against `pvlib.irradiance.get_total_irradiance` | relative 1e-9, absolute 1e-6 W/m² |
| `test_chain.py` | sky-view factor of a horizontal plane behind a uniform horizon h; beam with the sun below the horizon | cos² h; zero |
| `test_chain.py` | inverter loading ratio and site AC limit | clipping only above P_ac0; limit never exceeded |
| `test_chain.py` | snow state: zero tilt, 6 h trigger, depth reset; independence of the time step | error raised; Eqs. 4 to 6 of the article |
| `test_schema.py` | site files: unknown key or section, missing entry, value out of range, unknown provenance class | error naming the entry |
| `test_schema.py` | every assessment site file | valid, every parameter classified, U_c never from output |
| `test_schema.py` | datasheet De Soto fit at STC | I_sc, V_oc, P_mp within relative 1e-8 |
| `test_schema.py` | CEC path at STC; dV_oc/dT | CEC STC curve within relative 1e-6; datasheet β_oc within relative 1e-4 |
| `test_schema.py` | stored hourly RePV series, Las Vegas 34, 2011 (seven output columns) | absolute 1e-9 kW (or W/m², °C) |
| `test_bench.py` | stored hourly series of the reference chain; scores of the reference chain and RePV | absolute 1e-9 kW; relative 1e-8 |
| `test_bench.py` | each single-option variant | only the named option changes |
| `test_bench.py` | record with a known clock shift; outage-day removal | shift recovered; outage days dropped |
| `test_nrel.py` | NSRDB and MIDC quality control and hourly means | BSRN limits applied; hour means of valid steps |
| `test_era5_grid.py` | grid sampling and advection utilities (optional inputs, not used in the article) | analytical cases |

---

## 2. Software and reproducibility

- Package `solarmodel` 1.0.0, Python ≥ 3.12. The environment of every number in the article is `requirements-lock.txt`: Python 3.13, pvlib 0.16.1, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, pyarrow 26.0.1. The assessment outputs were first produced with pvlib 0.15.2; rerunning a site in the pinned environment reproduces its hourly series within 1e-14 kW.
- A model is selected by name with optional option overrides, e.g. `repv`, `repv:snow_use_depth=false`, `rating`.
- `python scripts/run_pvdaq_validation.py` runs every assessment site from `studies/PVDAQ_validation/validation.toml` and the weather cache; `latex/methodsx/make_validation_tables.py` then writes every table and number of the article, and `make_repo_supplement.py` this file.
- `python latex/methodsx/make_worked_example.py --site LasVegas_34 --year 2011` writes the worked example of Section S1.3.
- `python scripts/snow_depth_binding.py` counts the hours in which the ground-snow-depth condition changes the snow state (Section 3.5 of the article).
- `python scripts/site_provenance.py --check` reports the provenance classes of every assessment site file.
- Weather, horizon and PVGIS downloads are cached; a rerun on the cache needs no network access. NSRDB requests need an NREL API key in `config/secrets.toml` (template: `config/secrets.example.toml`), not needed with the cache.
- Bootstrap: circular blocks of {g('Block')} consecutive retained days, {g('Reps')} replicates per site, seed {g('Seed')}, paired across all series of a site.
- The archived release (`scripts/make_release.py`) contains the source code, schema, site files with provenance, the cached weather of the assessment windows, the extracted hourly measured series (PVDAQ, CC BY 4.0), the assessment outputs, the study files, the paper sources and the verification suite.

---

## 3. Implementation notes

### 3.1 Circumsolar component

The Reindl circumsolar part used for horizon masking and row shading is

$$G_{{d,cs}} = G_{{DHI}}\\,\\frac{{G_{{DNI}}}}{{I_0}}\\,\\frac{{\\max(\\cos\\theta,0)}}{{\\max(\\cos\\theta_z,\\,0.01745)}}.$$

The lower limit 0.01745 is cos 89°.

### 3.2 Snow

RePV passes the trailing 6 h snowfall sum (as an amount per step, so that pvlib's hourly rate equals the window sum), the ERA5-Land depth (forward-filled onto the weather index) and the plane-of-array irradiance to `pvlib.snow.coverage_nrel`. The depth condition and reset are pvlib's `snow_depth` and `threshold_depth` arguments. `scripts/snow_depth_binding.py` shows that, at the 17 independent sites with snowfall, the condition suppressed no event and changed the cover in at most 14 hours of a scored year.

### 3.3 Heat-loss coefficient

U_c came from the PVsyst mounting class at {g('UcClass')} sites, the datasheet NOCT at {g('UcNoct')} sites (Arbuckle 2107, Golden 4, Las Vegas 34) and a fit to eight module-back sensors at Henderson 1423 (`studies/LasVegas_34/fit_thermal.py`). The NOCT, defined at open circuit, gives U_c = α_abs · 800 / (T_NOCT − 20).

### 3.4 Module parameters

22 sites use the CEC path of Section S1.1 and 7 the datasheet fit. The solved band gap of the CEC path lies between 1.01 and 1.10 eV for the silicon modules and is 1.26 eV for the CIGS module of Golden 1289.

---

## 4. Extended results

### 4.1 Kling-Gupta efficiency

$$\\mathrm{{KGE}} = 1 - \\sqrt{{(r-1)^2 + (\\alpha-1)^2 + (\\beta-1)^2}},\\qquad \\alpha = \\sigma_s/\\sigma_m,\\quad \\beta = \\mu_s/\\mu_m,$$

the original form of Gupta et al. (2009), with r the Pearson correlation of simulated and measured hourly power on the scored hours.

### 4.2 Interpretation of the bootstrap comparisons

- For a change from series A to series B, the paired differences are Δ|b| = |b_A| − |b_B|, ΔNRMSE = NRMSE_A − NRMSE_B and ΔKGE = KGE_B − KGE_A; positive values favour B.
- B is closer to the measured output than A at a site when the 95 % paired interval of the difference lies above zero; A is closer when it lies below; otherwise the comparison is unresolved (U).
- Each metric is assessed separately, without adjustment for multiple comparisons. An excluded zero identifies a difference reproducible within the scored year, not one of practical size.
- The intervals describe within-year sampling variability conditional on site, model, weather product and quality-control rules, not site-to-site or interannual variability, parameter uncertainty or uncertainty at a new site.

### 4.3 Summary scores ({n} independent sites)

{summary}

### 4.4 Paired comparisons

{pairs}

### 4.5 One-at-a-time ablations

Change in signed bias Δb (percentage points) when a component is removed from RePV alone, and when it is added alone to the reference chain; sites with |Δb| ≥ 0.1. Positive Δb: the component raises the simulated energy. The complete snow model acts much less in the reference chain, where it is triggered by hourly snowfall, which rarely exceeds 1 cm in reanalysis.

**ERA5**

{abl['ERA5']}

**NSRDB**

{abl['NSRDB']}

### 4.6 Regional results (RePV signed bias, %)

{regions}

At Golden, station irradiance removes most of the ERA5 over-prediction, which places it in the ERA5 irradiance at the foot of the Front Range.

### 4.7 Station-irradiance subset ({g('NStation')} independent sites)

{station}

The residual NRMSE on station irradiance combines conversion error, spatial mismatch between station and array, ERA5 temperature error, metadata error and measurement error, so it bounds the conversion error from above.

### 4.8 Calibrated sites ({g('NCal')} sites)

{calibrated}

Per-site scores with bootstrap intervals for every series: Tables S11 and S12 of the journal supplement.

---

## 5. Items that only the author can complete

- Repository URL, Zenodo DOI (create the GitHub release, then archive it on Zenodo and enter both in the Specifications table and the Code availability statement).
- Co-authors and CRediT roles, acknowledgements and funding statement.
"""
    (HERE / "supplementary.md").write_text(text)
    print(f"-> {HERE / 'supplementary.md'}")


if __name__ == "__main__":
    main()
