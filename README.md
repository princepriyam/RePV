# RePV: reanalysis-driven photovoltaic yield model

[![tests](https://github.com/princepriyam/RePV/actions/workflows/tests.yml/badge.svg)](https://github.com/princepriyam/RePV/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24355624.svg)](https://doi.org/10.5281/zenodo.24355624)

RePV converts hourly weather (ERA5 by default, any irradiance source on request) and a description of a PV system into the mean AC power of one array at the grid connection point, with every modelling decision fixed and documented: the order of the published models, the source of each parameter value, the time interval each value represents, the starting state of snow and soiling, the default used when an input is missing, and whether each loss acts before or after the inverter. The method, its validation against 25 NREL PVDAQ systems and its limitations are described in the MethodsX article.

RePV is the model `repv` of the Python package `solarmodel`.

## Install

```
python -m pip install -r requirements-lock.txt    # the environment of every number in the article
python -m pip install -e .                        # the package
python -m pytest                                  # verification suite
```

`requirements-lock.txt` pins Python 3.13 packages (pvlib 0.16.1, pandas 3.0.5, NumPy 2.5.3). `pyproject.toml` gives the minimum versions.

The weather cache, measured series and per-site outputs are not in this repository; they come from the Zenodo archive (https://doi.org/10.5281/zenodo.24355624).

## Run RePV on one site

Offline, on the Las Vegas test year shipped with the tests:

```
python examples/run_repv.py                       # writes examples/repv_LasVegas_34_2011.csv
```

On any configured site, downloading ERA5 from the Open-Meteo archive (free, no key) on first use and caching it:

```
python -m solarmodel.bench --sites LasVegas_34 --models repv rating
```

From Python:

```python
from solarmodel import bench, models
from solarmodel.weather import inputs

site = bench.load_site("LasVegas_34")                       # config/site/LasVegas_34.toml
m = models.repv()                                           # or models.parse("repv:snow_use_depth=false")
w = inputs.weather("point", "point", site.name, site.system, site.start, site.end, {}, m)
out = bench.simulate(site, m, w)                            # p_ac_kw, p_dc_kw, poa_global, poa_eff, t_cell, snow_tx, soiling
```

## Site files

A site is one TOML file in `config/site/`. Its sections, required entries, ranges and the six provenance classes are defined in `solarmodel/schema.py`; a missing required entry, an unknown key or section, a value out of range or an unknown provenance class stops a run with an error that names the entry. Every model parameter of an assessment site carries its source class in the `[provenance]` table (`python scripts/site_provenance.py --check`).

## Reproduce the validation

```
python scripts/run_pvdaq_validation.py            # every assessment site, from the weather cache
```

NSRDB irradiance needs an NREL API key in `config/secrets.toml` (see `config/secrets.example.toml`) unless the cache holds it; the archived release ships the cache of every assessment window.

## Layout

| Path | Content |
| --- | --- |
| `solarmodel/` | the chain (`model/chain.py`), named models (`models.py`), site-file schema (`schema.py`), weather inputs, testbench |
| `config/site/` | site files of the assessment sites, with provenance tables |
| `studies/PVDAQ_validation/` | the study file of the article's validation |
| `scripts/` | PVDAQ extraction, site-file writer, provenance, validation runner, snow-depth count, Zenodo archive builder, DOI setter |
| `outputs/pvdaq_validation/` | assessment outputs (hourly series, scores, bootstrap) |
| `tests/` | verification suite (run on every push, `.github/workflows/tests.yml`) |
| `studies/LasVegas_34/fit_thermal.py` | U_c fitted to module-temperature sensors (Henderson 1423) |

## Licence and citation

Code: MIT licence (`LICENSE`). Measured PV data: NREL PVDAQ, CC BY 4.0. Weather: ERA5 and ERA5-Land (Copernicus licence) through Open-Meteo (CC BY 4.0); NSRDB and MIDC data from NREL; PVGIS from the European Commission JRC. Cite the article and the archived release (`CITATION.cff`).
