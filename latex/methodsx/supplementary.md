# RePV: repository supplement to the MethodsX article

This file holds material that is not part of the article (`main.tex`) or its journal supplement (`supplement.tex`): the verification tests with their acceptance criteria, the reproducibility settings, implementation notes and the extended results. Table and section numbers prefixed S refer to the journal supplement; all other numbers refer to the article. It is written by `latex/methodsx/make_repo_supplement.py` from `val_numbers.tex`; do not edit it by hand.

**Names.** RePV is the model `repv` of the Python package `solarmodel` (`solarmodel.models.repv()`, identical to `current()`). The simple rating model of the validation is `rating` (identical to `homer()`). The reference chain of the ablations is `thesis`: RePV without site-pressure air mass, diffuse IAM, terrain horizon, physical loss placement, soiling on the irradiance, AC capacity limit, ground-snow-depth condition, 6 h trigger and snow albedo.

**Terminology.** *Verification*: tests of the implementation against reference implementations, limiting cases and stored outputs (Section 1). *Independent validation*: the 25 sites configured without reference to measured output. *Calibrated assessment*: the 4 Mid-Atlantic sites whose fast snow removal was set after comparison with measured output. *Reference model*: PVGIS-ERA5, an independently developed model, not an observational reference.

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
- Bootstrap: circular blocks of 5 consecutive retained days, 2,000 replicates per site, seed 20261109, paired across all series of a site.
- The archived release (`scripts/make_release.py`) contains the source code, schema, site files with provenance, the cached weather of the assessment windows, the extracted hourly measured series (PVDAQ, CC BY 4.0), the assessment outputs, the study files, the paper sources and the verification suite.

---

## 3. Implementation notes

### 3.1 Circumsolar component

The Reindl circumsolar part used for horizon masking and row shading is

$$G_{d,cs} = G_{DHI}\,\frac{G_{DNI}}{I_0}\,\frac{\max(\cos\theta,0)}{\max(\cos\theta_z,\,0.01745)}.$$

The lower limit 0.01745 is cos 89°.

### 3.2 Snow

RePV passes the trailing 6 h snowfall sum (as an amount per step, so that pvlib's hourly rate equals the window sum), the ERA5-Land depth (forward-filled onto the weather index) and the plane-of-array irradiance to `pvlib.snow.coverage_nrel`. The depth condition and reset are pvlib's `snow_depth` and `threshold_depth` arguments. `scripts/snow_depth_binding.py` shows that, at the 17 independent sites with snowfall, the condition suppressed no event and changed the cover in at most 14 hours of a scored year.

### 3.3 Heat-loss coefficient

U_c came from the PVsyst mounting class at 25 sites, the datasheet NOCT at 3 sites (Arbuckle 2107, Golden 4, Las Vegas 34) and a fit to eight module-back sensors at Henderson 1423 (`studies/LasVegas_34/fit_thermal.py`). The NOCT, defined at open circuit, gives U_c = α_abs · 800 / (T_NOCT − 20).

### 3.4 Module parameters

22 sites use the CEC path of Section S1.1 and 7 the datasheet fit. The solved band gap of the CEC path lies between 1.01 and 1.10 eV for the silicon modules and is 1.26 eV for the CIGS module of Golden 1289.

---

## 4. Extended results

### 4.1 Kling-Gupta efficiency

$$\mathrm{KGE} = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2},\qquad \alpha = \sigma_s/\sigma_m,\quad \beta = \mu_s/\mu_m,$$

the original form of Gupta et al. (2009), with r the Pearson correlation of simulated and measured hourly power on the scored hours.

### 4.2 Interpretation of the bootstrap comparisons

- For a change from series A to series B, the paired differences are Δ|b| = |b_A| − |b_B|, ΔNRMSE = NRMSE_A − NRMSE_B and ΔKGE = KGE_B − KGE_A; positive values favour B.
- B is closer to the measured output than A at a site when the 95 % paired interval of the difference lies above zero; A is closer when it lies below; otherwise the comparison is unresolved (U).
- Each metric is assessed separately, without adjustment for multiple comparisons. An excluded zero identifies a difference reproducible within the scored year, not one of practical size.
- The intervals describe within-year sampling variability conditional on site, model, weather product and quality-control rules, not site-to-site or interannual variability, parameter uncertainty or uncertainty at a new site.

### 4.3 Summary scores (25 independent sites)

| Series | Irradiance | Median b (%) | Median \|b\| (%) | Sites \|b\| ≤ 5 % | Sites \|b\| ≤ 10 % | Median NRMSE (%) | Median KGE |
|---|---|---|---|---|---|---|---|
| Rating | ERA5 | +15.3 | 15.3 | 3 | 6 | 38.6 | 0.79 |
| Rating | NSRDB | +12.3 | 12.3 | 5 | 11 | 29.2 | 0.84 |
| RePV | ERA5 | +6.7 | 6.7 | 12 | 16 | 37.4 | 0.84 |
| RePV | NSRDB | +2.6 | 6.0 | 10 | 18 | 27.5 | 0.87 |
| PVGIS | PVGIS-ERA5 | +6.5 | 6.5 | 11 | 15 | 38.2 | 0.82 |

### 4.4 Paired comparisons

| Change A → B | n | \|b\|: B / A / U | NRMSE: B / A / U | KGE: B / A / U | Median Δ\|b\| | Median ΔNRMSE |
|---|---|---|---|---|---|---|
| RePV ERA5 → RePV NSRDB | 25 | 10 / 7 / 8 | 25 / 0 / 0 | 21 / 0 / 4 | +0.8 | +7.8 |
| RePV NSRDB → RePV station | 16 | 5 / 3 / 8 | 10 / 2 / 4 | 11 / 1 / 4 | +0.4 | +2.5 |
| RePV ERA5 → PVGIS-ERA5 | 25 | 5 / 9 / 11 | 8 / 6 / 11 | 9 / 9 / 7 | −0.5 | −0.3 |
| PVGIS-ERA5 → RePV NSRDB | 25 | 10 / 6 / 9 | 21 / 0 / 4 | 17 / 4 / 4 | +0.3 | +7.7 |

### 4.5 One-at-a-time ablations

Change in signed bias Δb (percentage points) when a component is removed from RePV alone, and when it is added alone to the reference chain; sites with |Δb| ≥ 0.1. Positive Δb: the component raises the simulated energy. The complete snow model acts much less in the reference chain, where it is triggered by hourly snowfall, which rarely exceeds 1 cm in reanalysis.

**ERA5**

| Component | Removed from RePV: median Δb | Range | Sites | Added to the reference chain: median Δb | Range | Sites |
|---|---|---|---|---|---|---|
| Site-pressure air mass | −0.32 | −1.1 to 0.0 | 18/25 | −0.32 | −1.0 to 0.0 | 18/25 |
| Diffuse IAM | −1.69 | −2.1 to −1.0 | 25/25 | −1.55 | −2.1 to −1.0 | 25/25 |
| Terrain horizon | 0.00 | 0.0 to 0.0 | 0/25 | 0.00 | 0.0 to 0.0 | 0/25 |
| Physical loss placement | −0.02 | −0.1 to +0.4 | 6/25 | −0.01 | −0.4 to +0.6 | 9/25 |
| Soiling on irradiance | 0.00 | −0.2 to +0.3 | 5/25 | +0.01 | −0.2 to +0.2 | 5/25 |
| AC capacity limit | 0.00 | −0.2 to 0.0 | 1/25 | 0.00 | −0.5 to 0.0 | 2/25 |
| Snow-depth condition and reset | 0.00 | 0.0 to 0.0 | 0/25 | 0.00 | 0.0 to 0.0 | 0/25 |
| 6 h snowfall trigger | −1.75 | −7.6 to 0.0 | 17/25 | −1.76 | −7.8 to 0.0 | 17/25 |
| Snow-dependent albedo | +0.03 | 0.0 to +1.5 | 8/25 | +0.10 | 0.0 to +1.6 | 12/25 |
| Complete snow model | −2.08 | −17.5 to 0.0 | 17/25 | −0.11 | −10.0 to 0.0 | 13/25 |

**NSRDB**

| Component | Removed from RePV: median Δb | Range | Sites | Added to the reference chain: median Δb | Range | Sites |
|---|---|---|---|---|---|---|
| Site-pressure air mass | −0.31 | −1.0 to 0.0 | 18/25 | −0.31 | −0.9 to 0.0 | 18/25 |
| Diffuse IAM | −1.86 | −2.4 to −1.1 | 25/25 | −1.70 | −2.5 to −1.0 | 25/25 |
| Terrain horizon | 0.00 | 0.0 to 0.0 | 0/25 | 0.00 | 0.0 to 0.0 | 0/25 |
| Physical loss placement | −0.02 | −0.1 to +0.3 | 4/25 | −0.01 | −0.5 to +0.4 | 6/25 |
| Soiling on irradiance | 0.00 | −0.2 to +0.3 | 4/25 | 0.00 | −0.2 to +0.2 | 4/25 |
| AC capacity limit | 0.00 | −0.1 to 0.0 | 1/25 | 0.00 | −0.7 to 0.0 | 3/25 |
| Snow-depth condition and reset | 0.00 | 0.0 to 0.0 | 0/25 | 0.00 | 0.0 to 0.0 | 0/25 |
| 6 h snowfall trigger | −1.39 | −6.0 to 0.0 | 17/25 | −1.41 | −6.1 to 0.0 | 17/25 |
| Snow-dependent albedo | +0.02 | 0.0 to +1.4 | 8/25 | +0.07 | 0.0 to +1.5 | 10/25 |
| Complete snow model | −2.11 | −15.8 to 0.0 | 17/25 | −0.10 | −10.2 to 0.0 | 13/25 |

### 4.6 Regional results (RePV signed bias, %)

| Region | Sites | ERA5 median (range) | NSRDB median (range) | Station median (range) | PVGIS-ERA5 range |
|---|---|---|---|---|---|
| Golden, Colorado Front Range | 10 | +20.1 (−1.6 to +27.2) | +5.8 (−11.7 to +12.1) | +5.0 (−11.1 to +13.7) | −2.5 to +36.3 |
| Southern Nevada | 6 | −0.3 (−4.1 to +3.7) | −1.6 (−6.7 to +2.6) | −2.3 (−6.3 to +1.1) | −1.1 to +4.5 |
| East of the Mississippi | 7 | +3.4 (−3.9 to +16.7) | +8.4 (−3.1 to +13.6) | n/a | −3.6 to +13.3 |

At Golden, station irradiance removes most of the ERA5 over-prediction, which places it in the ERA5 irradiance at the foot of the Front Range.

### 4.7 Station-irradiance subset (16 independent sites)

| Series | Irradiance | Median b (%) | Median \|b\| (%) | Median NRMSE (%) | Median KGE |
|---|---|---|---|---|---|
| Rating | NSRDB | +10.1 | 10.1 | 31.3 | 0.84 |
| RePV | ERA5 | +11.0 | 11.0 | 39.5 | 0.79 |
| RePV | NSRDB | +0.9 | 5.3 | 30.0 | 0.86 |
| RePV | station | −0.5 | 3.8 | 22.0 | 0.89 |
| PVGIS | PVGIS-ERA5 | +12.6 | 12.6 | 40.3 | 0.75 |

The residual NRMSE on station irradiance combines conversion error, spatial mismatch between station and array, ERA5 temperature error, metadata error and measurement error, so it bounds the conversion error from above.

### 4.8 Calibrated sites (4 sites)

| Series | Irradiance | Median b (%) | Median \|b\| (%) | Median NRMSE (%) | Median KGE |
|---|---|---|---|---|---|
| Rating | NSRDB | +13.8 | 13.8 | 28.1 | 0.84 |
| RePV | ERA5 | +4.6 | 4.6 | 35.1 | 0.85 |
| RePV | NSRDB | +7.0 | 7.0 | 24.9 | 0.90 |
| PVGIS | PVGIS-ERA5 | +2.6 | 4.3 | 36.2 | 0.85 |

Per-site scores with bootstrap intervals for every series: Tables S11 and S12 of the journal supplement.

---

## 5. Items that only the author can complete

- Repository URL, Zenodo DOI (create the GitHub release, then archive it on Zenodo and enter both in the Specifications table and the Code availability statement).
- Co-authors and CRediT roles, acknowledgements and funding statement.
