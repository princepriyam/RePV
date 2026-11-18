# solarmodel

PV power from weather. One chain (`model/chain.py::run_chain`), named models (`repv`, `rating`, and their working names `current`, `homer`; `thesis`), one testbench. RePV, the method of the MethodsX article, is `models.repv()`.

| Model | Use |
| --- | --- |
| `models.thesis()` | The reference chain. Reproduces the stored validation series (`tests/test_bench.py`). |
| `models.repv()` = `models.current()` | RePV, the reviewed chain: site pressure, terrain horizon, row shading, IAM on diffuse light, snow with ground depth, physical loss order with AC rating, optical soiling. |
| `models.rating()` = `models.homer()` | The simple rating model of the article: NOCT cell temperature and PVWatts DC on the reference chain's POA; bench only. |

Change any option for one run: `models.current(iam="none")`, `models.thesis(snow="none", ac_clip=True)`. Every keyword is a `ChainOptions` field (see `model/chain.py`); a wrong name or value raises.

## Testbench

```
python -m solarmodel.bench                                     # every site with [measured]: reference chain vs RePV
python -m solarmodel.bench --sites Henderson_1423 --models thesis current "current:iam=none"
```

A study is a TOML file in a study folder: sites, window, models, irradiance inputs (`point`, ERA5 grid samples, advected grid), one temperature/wind source, MOS corrections, and breakdowns by month, hour, clear-sky index or temperature. `python -m solarmodel.bench --study studies/LasVegas_34/inputs.toml` writes to `studies/LasVegas_34/outputs/inputs/`. Scoring drops outage days and scores every series on the same clean daylight hours.

Writes `outputs/bench/metrics.csv` (bias, NRMSE, KGE, energy and specific yield per site and model) and `hourly.csv.gz`. All models on a site see the same weather and are scored on the same hours. From Python: `bench.run(sites, models)`.

## Files

| File | Purpose |
| --- | --- |
| `models.py` | `repv()`, `rating()`, `thesis()`, `current()`, `parse("repv:iam=none")` |
| `schema.py` | site-file format, ranges and the six provenance classes; checked before every run |
| `bench.py` | Site loading, measured-data alignment, scoring, CLI |
| `model/chain.py` | `ChainOptions`, `PVSystem` (`from_site_toml`), `run_chain`, stage functions, `module_report` |
| `model/transposition.py` | POA: pvlib sky models plus Muneer (PVGIS), horizon blocking and sky view, rows via infinite sheds |
| `model/power.py` | De Soto fit from datasheet values, SAPM spectral factor, loss stack |
| `losses/snow.py` | NREL snow coverage with ground depth and rolling trigger, snow albedo |
| `losses/soiling_loss.py` | Kimber soiling, fixed loss |
| `weather/utils.py` | Open-Meteo downloads with a parquet cache; gaps raise |
| `weather/site.py` | One variable per call; `hourly_weather` assembles the chain input |
| `weather/horizon.py` | PVGIS horizon (cached), merged profiles |
| `weather/inputs.py` | Irradiance and temperature/wind inputs for the bench, as spec strings; `weather()` joins one input for one model |
| `weather/nrel.py` | MIDC station data and NSRDB PSM v4: QC, 10-min steps and hour means, cached |
| `weather/pvgis.py` | PVGIS-ERA5 PV output as a benchmark series; the shared PVGIS download and HH:30 label rule |
| `weather/cache.py` | Parquet read/write with a checked conversion of legacy CSV caches |
| `weather/era5_grid.py` | ERA5 grid loading, site sampling (nearest, bilinear, Gaussian), steering wind, advection (`[nwp]` extra) |
| `weather/mos.py` | Neighbourhood MOS on the PV clear-sky index, leave one year out (`[nwp]` extra) |

Site configs: `config/site/<site>.toml`, one per site, with the scoring window in `[data].sa_start`/`sa_end`.

Grid downloads, MERRA-2, site adaptation, station loading, calibration and the older scripts are on the `nwp-study` branch (tag `v0.3.1-pre-cleanup`).
