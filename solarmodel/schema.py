"""Checks of a site file (``config/site/<name>.toml``) before a run.

:func:`validate_site` is called by :meth:`solarmodel.model.chain.PVSystem.from_site_toml`,
so every run stops on a malformed site file. It reports every problem at once:

* a required entry that is missing;
* a section or key that is not part of the format (a misspelt optional key would
  otherwise fall back to its default without notice);
* a value of the wrong type or outside its physical range;
* a ``[provenance]`` entry that names an unknown class or a key the file does not hold.

Provenance. Every model parameter of an assessment site carries one of six source
classes in the ``[provenance]`` table, keyed ``"<section>.<key>"``:

========== =====================================================================
measured   measured on site (e.g. U_c fitted to module-temperature sensors)
datasheet  manufacturer datasheet, or the CEC module or inverter database entry
published  published default (Kimber soiling, PVsyst U-values, De Soto glass, ...)
metadata   inferred from the system metadata (location, orientation, counts, age)
output     inferred from measured AC output (fast snow removal at calibrated sites)
assumed    assumed, with no site-specific source
========== =====================================================================

:func:`missing_provenance` lists the parameters without a class; the test suite
requires it to be empty for every assessment site.
"""
from __future__ import annotations

import math

PROVENANCE_CLASSES = {
    "measured": "measured on site",
    "datasheet": "manufacturer datasheet or database entry",
    "published": "published default",
    "metadata": "inferred from system metadata",
    "output": "inferred from measured output",
    "assumed": "assumed",
}

# section -> {key: (kind, low, high)}; kind: "num", "int", "str", "bool", "list"; bounds inclusive, None = open
_NUM = "num"
SCHEMA: dict[str, dict[str, tuple]] = {
    "location": {"name": ("str", None, None), "lat": (_NUM, -90, 90), "lon": (_NUM, -180, 180),
                 "altitude": (_NUM, -500, 9000)},
    "array": {"tracking": ("bool", None, None), "tilt": (_NUM, 0, 90), "azimuth": (_NUM, 0, 360),
              "albedo": (_NUM, 0, 1), "snow_tilt": (_NUM, 0, 90)},
    "system": {"rated_capacity_kw": (_NUM, 0, None), "pdc0_w": (_NUM, 0, None), "ac_cap_kw": (_NUM, 0, None),
               "ns_arr": ("int", 1, None), "np_arr": ("int", 1, None)},
    "module": {"isc_ref": (_NUM, 0, None), "voc_ref": (_NUM, 0, None), "imp_ref": (_NUM, 0, None),
               "vmp_ref": (_NUM, 0, None), "mu_isc": (_NUM, -1, 1), "mu_voc": (_NUM, -5, 0),
               "ns": ("int", 1, None), "alpha_p": (_NUM, -0.02, 0), "eta_mp_stc": (_NUM, 0, 0.5),
               "eg_ref": (_NUM, 0.5, 2.5), "cec": ("str", None, None), "bifaciality": (_NUM, 0, 1),
               "spectral": ("str", None, None), "huld_cell_type": ("str", None, None)},
    "iam": {"n": (_NUM, 1, 3), "k": (_NUM, 0, None), "l": (_NUM, 0, 0.1), "a_r": (_NUM, 0, 1)},
    "thermal.pvsyst": {"u_c": (_NUM, 0, 100), "u_v": (_NUM, 0, 20), "eta_module": (_NUM, 0, 0.5),
                       "alpha_abs": (_NUM, 0, 1)},
    "thermal.faiman": {"u0": (_NUM, 0, 100), "u1": (_NUM, 0, 20)},
    "thermal.homer": {"tc_noct": (_NUM, 20, 80), "ta_noct": (_NUM, -20, 50), "poa_noct": (_NUM, 0, 1.5),
                      "tau_alpha": (_NUM, 0, 1)},
    "losses": {"n_degradation": (_NUM, 0, 1), "n_mismatch": (_NUM, 0, 1), "n_dc_wiring": (_NUM, 0, 1),
               "n_ac_wiring": (_NUM, 0, 1), "n_inverter": (_NUM, 0, 1), "n_temp": (_NUM, 0, 1),
               "n_soiling": (_NUM, 0, 1), "f_pv": (_NUM, 0, 1), "eta_inv": (_NUM, 0, 1)},
    "soiling": {"model": ("str", None, None), "loss_rate": (_NUM, 0, 0.1), "cleaning_threshold_mm": (_NUM, 0, None),
                "grace_period_days": ("int", 0, None), "max_soiling": (_NUM, 0, 1),
                "manual_cleaning_dates": ("list", None, None), "fixed_loss": (_NUM, 0, 1)},
    "inverter": {"cec": ("str", None, None), "count": ("int", 1, None), "ilr": (_NUM, 0.5, 3)},
    "layout": {"gcr": (_NUM, 0, 1), "height_m": (_NUM, 0, None), "collector_width_m": (_NUM, 0, None)},
    "tracker": {"axis_tilt": (_NUM, 0, 90), "axis_azimuth": (_NUM, 0, 360), "max_angle": (_NUM, 0, 90),
                "backtrack": ("bool", None, None), "gcr": (_NUM, 0, 1), "cross_axis_tilt": (_NUM, -90, 90)},
    "snow": {"removal": ("str", None, None)},
}
# sections that describe data handling, not the PV system: any key is accepted
FREE_SECTIONS = {"data", "measured", "station", "geothermal"}
REQUIRED = {
    "location": ("lat", "lon", "altitude"),
    "array": ("tilt", "azimuth", "albedo"),
    "system": ("ns_arr", "np_arr"),
    "module": ("isc_ref", "voc_ref", "imp_ref", "vmp_ref", "mu_isc", "mu_voc", "ns"),
    "iam": ("n", "k", "l"),
    "thermal.pvsyst": ("u_c", "u_v", "eta_module", "alpha_abs"),
    "losses": ("n_mismatch", "n_dc_wiring", "n_ac_wiring", "n_inverter"),
}
CHOICES = {("soiling", "model"): {"kimber", "fixed", "none"}, ("snow", "removal"): {"fast", "sliding"}}
# keys that are not model parameters and need no provenance
_NO_PROVENANCE = {("location", "name")}


class SiteConfigError(ValueError):
    """A site file that does not satisfy the format; the message lists every problem."""


def _sections(site: dict) -> dict[str, dict]:
    """``{"thermal.pvsyst": {...}, "array": {...}, ...}``: nested tables flattened one level."""
    out = {}
    for name, val in site.items():
        if name == "thermal" and isinstance(val, dict):
            for sub, tab in val.items():
                out[f"thermal.{sub}"] = tab
        else:
            out[name] = val
    return out


def _check_value(where: str, v, kind: str, lo, hi) -> str | None:
    if kind == "str":
        return None if isinstance(v, str) else f"{where}: expected text, got {v!r}"
    if kind == "bool":
        return None if isinstance(v, bool) else f"{where}: expected true/false, got {v!r}"
    if kind == "list":
        return None if isinstance(v, list) else f"{where}: expected a list, got {v!r}"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return f"{where}: expected a number, got {v!r}"
    if kind == "int" and float(v) != int(v):
        return f"{where}: expected a whole number, got {v!r}"
    if not math.isfinite(float(v)):
        return f"{where}: not finite"
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return f"{where} = {v} outside [{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}]"
    return None


def parameter_keys(site: dict) -> list[str]:
    """The model parameters of a site file, as ``"<section>.<key>"``."""
    out = []
    for sec, tab in _sections(site).items():
        if sec in SCHEMA and isinstance(tab, dict):
            out += [f"{sec}.{k}" for k in tab if (sec, k) not in _NO_PROVENANCE]
    return out


def validate_site(site: dict) -> None:
    """Raise :class:`SiteConfigError` listing every problem of a parsed site file."""
    errs = []
    secs = _sections(site)
    for sec, tab in secs.items():
        if sec == "provenance" or sec in FREE_SECTIONS:
            continue
        if sec not in SCHEMA:
            errs.append(f"[{sec}]: unknown section (allowed: {', '.join(sorted(SCHEMA))}, "
                        f"{', '.join(sorted(FREE_SECTIONS))}, provenance)")
            continue
        if not isinstance(tab, dict):
            errs.append(f"[{sec}]: expected a table")
            continue
        for k, v in tab.items():
            spec = SCHEMA[sec].get(k)
            if spec is None:
                errs.append(f"{sec}.{k}: unknown key (allowed: {', '.join(sorted(SCHEMA[sec]))})")
                continue
            e = _check_value(f"{sec}.{k}", v, *spec)
            if e:
                errs.append(e)
            elif (sec, k) in CHOICES and v not in CHOICES[(sec, k)]:
                errs.append(f"{sec}.{k} = {v!r}: choose one of {sorted(CHOICES[(sec, k)])}")
    for sec, keys in REQUIRED.items():
        tab = secs.get(sec)
        if not isinstance(tab, dict):
            errs.append(f"[{sec}]: required section missing")
            continue
        errs += [f"{sec}.{k}: required entry missing" for k in keys if k not in tab]
    if secs.get("array", {}).get("tracking", False) and "tracker" not in secs:
        errs.append("[tracker]: required when array.tracking = true")
    prov = site.get("provenance", {})
    if not isinstance(prov, dict):
        errs.append("[provenance]: expected a table of \"section.key\" = class")
    else:
        present = set(parameter_keys(site))
        for k, c in prov.items():
            if c not in PROVENANCE_CLASSES:
                errs.append(f"provenance.{k} = {c!r}: choose one of {sorted(PROVENANCE_CLASSES)}")
            if k not in present:
                errs.append(f"provenance.{k}: no such parameter in this file")
    if errs:
        raise SiteConfigError("site file: " + "; ".join(errs))


def missing_provenance(site: dict) -> list[str]:
    """Model parameters of ``site`` that have no provenance class."""
    prov = site.get("provenance", {}) or {}
    return [k for k in parameter_keys(site) if k not in prov]
