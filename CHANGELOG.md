# Changelog

## 1.0.0 (18 Nov 2026), MethodsX submission

Added
- `models.repv()` and `models.rating()`: RePV and the simple rating model under their article names (identical to `current()` and `homer()`).
- `solarmodel/schema.py`: site-file check before every run (required entries, unknown sections and keys, ranges, provenance classes); `SiteConfigError` names every offending entry.
- `[provenance]` table in every PVDAQ assessment site file, one of six classes per parameter (`scripts/site_provenance.py`; `pvdaq_site_configs.py` writes it).
- `tests/test_schema.py`: malformed site files, provenance completeness, De Soto STC reproduction (datasheet fit 1e-8, CEC path 1e-6 and dVoc/dT 1e-4), stored hourly RePV series (`tests/data/bench/repv_series.csv`).
- `scripts/snow_depth_binding.py`, `scripts/make_release.py`, `examples/run_repv.py`, `README.md`, `LICENSE` (MIT), `CITATION.cff`, `requirements-lock.txt`.
- `latex/methodsx/make_repo_supplement.py` writes `supplementary.md` from `val_numbers.tex`.
- `.github/workflows/tests.yml`: the verification suite and the offline example on every push (Python 3.13, pinned environment).
- `scripts/set_doi.py`: writes the Zenodo DOI into the article, `CITATION.cff`, `README.md` and `pyproject.toml`.

Changed
- `config/site/{Arbuckle_2107,Golden_4,LasVegas_34}.toml`: U_c from the datasheet NOCT without the (1 - eta) factor, because NOCT is measured at open circuit: 22.9 -> 27.2, 23.8 -> 28.8, 24.2 -> 28.3 W/m2K. Those three sites were rerun; the median bias and NRMSE of every series are unchanged, and 11 numbers quoted in the article changed (largest: median bias at Golden on station irradiance, +4.3 -> +5.0 %).
- `chain._diffuse_iam`: accepts the dict that `pvlib.iam.martin_ruiz_diffuse` returns from pvlib 0.16.
- `make_validation_tables.py`, `make_worked_example.py`: RePV names; NaN station shown as n/a with pandas 3.
- `studies/LasVegas_34/fit_thermal.py`: runs on its own (RePV on the cached weather; raw sensor files from the OEDI PVDAQ data lake).
- `scripts/make_release.py` builds the Zenodo archive (code, paper and data) and leaves basemap tiles out.
