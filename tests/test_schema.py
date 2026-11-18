"""Site-file checks, module-parameter extraction at STC and the stored RePV series.

Acceptance criteria are those of the repository supplement (latex/methodsx/supplementary.md,
Section 2): STC reproduction within relative 1e-8 for the datasheet fit, the CEC STC curve
within relative 1e-6, dVoc/dT of the CEC path within relative 1e-4 of the datasheet value,
and the stored hourly RePV series within 1e-9 kW.
"""
from __future__ import annotations

import copy
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import pytest

from solarmodel import bench, models, paths
from solarmodel.model.chain import PVSystem
from solarmodel.model.power import cec_module_params, fit_module_params
from solarmodel.schema import (PROVENANCE_CLASSES, SiteConfigError, missing_provenance,
                               validate_site)
from solarmodel.weather import inputs

DATA = Path(__file__).parent / "data" / "bench"
SITE = DATA / "LasVegas_34.toml"
SITE_DIR = paths.CONFIG_DIR / "site"


def _site() -> dict:
    return tomllib.loads(SITE.read_text())


def _stc(p: dict, t: float = 25.0) -> dict:
    a = pvlib.pvsystem.calcparams_desoto(1000.0, t, p["alpha_sc"], p["a_ref"], p["I_L_ref"], p["I_o_ref"],
                                         p["R_sh_ref"], p["R_s"], EgRef=p["EgRef"], dEgdT=p["dEgdT"])
    return {k: float(v) for k, v in pvlib.pvsystem.singlediode(*a).items()}


# ── site-file format ──────────────────────────────────────────────────────────

def test_test_site_is_valid():
    validate_site(_site())
    PVSystem.from_site_toml(_site())


@pytest.mark.parametrize("edit, message", [
    (lambda s: s["thermal"]["pvsyst"].__setitem__("u_cc", s["thermal"]["pvsyst"].pop("u_c")), "thermal.pvsyst.u_cc: unknown key"),
    (lambda s: s["array"].__setitem__("snow_tilit", 20.0), "array.snow_tilit: unknown key"),
    (lambda s: s["losses"].pop("n_mismatch"), "losses.n_mismatch: required entry missing"),
    (lambda s: s["array"].__setitem__("tilt", 95.0), "array.tilt = 95.0 outside"),
    (lambda s: s["array"].__setitem__("albedo", "0.2"), "array.albedo: expected a number"),
    (lambda s: s.__setitem__("arrray", {"tilt": 10.0}), "[arrray]: unknown section"),
    (lambda s: s.__setitem__("provenance", {"array.tilt": "guessed"}), "provenance.array.tilt = 'guessed'"),
    (lambda s: s.__setitem__("provenance", {"array.tilit": "metadata"}), "provenance.array.tilit: no such parameter"),
])
def test_malformed_site_is_rejected_with_its_name(edit, message):
    s = copy.deepcopy(_site())
    edit(s)
    with pytest.raises(SiteConfigError, match=None) as e:
        PVSystem.from_site_toml(s)
    assert message in str(e.value)


def _assessment_sites() -> list[Path]:
    if not SITE_DIR.exists():
        return []
    out = []
    for f in sorted(SITE_DIR.glob("*.toml")):
        m = tomllib.loads(f.read_text()).get("measured", {})
        if str(m.get("file", "")).startswith("data/validation/PVDAQ/"):
            out.append(f)
    return out


@pytest.mark.skipif(not _assessment_sites(), reason="no PVDAQ site files in config/site")
@pytest.mark.parametrize("path", _assessment_sites(), ids=lambda p: p.stem)
def test_assessment_site_is_valid_and_fully_classified(path):
    s = tomllib.loads(path.read_text())
    validate_site(s)
    assert missing_provenance(s) == []
    assert set(s["provenance"].values()) <= set(PROVENANCE_CLASSES)
    # U_c never comes from measured output (Section 2.6 of the article)
    assert s["provenance"]["thermal.pvsyst.u_c"] != "output"


# ── module parameters at STC ──────────────────────────────────────────────────

def test_datasheet_fit_reproduces_stc():
    ds = dict(Isc_ref=8.79, Voc_ref=37.0, Imp_ref=7.85, Vmp_ref=30.6, mu_Isc=0.00387, mu_Voc=-0.1217, Ns=60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd = _stc(fit_module_params(**ds))
    assert sd["i_sc"] == pytest.approx(ds["Isc_ref"], rel=1e-8)
    assert sd["v_oc"] == pytest.approx(ds["Voc_ref"], rel=1e-8)
    assert sd["p_mp"] == pytest.approx(ds["Imp_ref"] * ds["Vmp_ref"], rel=1e-8)


def test_cec_path_reproduces_cec_stc_and_datasheet_beta():
    name, mu_isc, mu_voc = "Sharp_NU_U240F1", 0.003806, -0.123046
    p = cec_module_params(name, mu_isc, mu_voc)
    db = pvlib.pvsystem.retrieve_sam("CECMod")[name]
    sd = _stc(p)
    assert sd["i_sc"] == pytest.approx(db["I_sc_ref"], rel=1e-6)
    assert sd["v_oc"] == pytest.approx(db["V_oc_ref"], rel=1e-6)
    assert sd["p_mp"] == pytest.approx(db["I_mp_ref"] * db["V_mp_ref"], rel=1e-6)
    dvoc = _stc(p, 25.5)["v_oc"] - _stc(p, 24.5)["v_oc"]
    assert dvoc == pytest.approx(mu_voc, rel=1e-4)
    assert 0.5 < p["EgRef"] < 2.5 and p["alpha_sc"] == mu_isc


# ── stored hourly RePV series ─────────────────────────────────────────────────

@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setattr(paths, "CACHE_DIR", DATA)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_repv_reproduces_stored_series(cache):
    site = bench.load_site(SITE)
    m = models.repv()
    w = inputs.weather("point", "point", site.name, site.system, site.start, site.end, {}, m)
    out = bench.simulate(site, m, w)
    stored = pd.read_csv(DATA / "repv_series.csv", parse_dates=["time"], index_col="time")
    stored.index = stored.index.tz_localize("UTC")
    for col in stored.columns:
        np.testing.assert_allclose(out[col].reindex(stored.index), stored[col], rtol=1e-11, atol=1e-9, err_msg=col)
