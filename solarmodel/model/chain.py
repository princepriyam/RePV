"""The PV model chain, in one place, with every modelling choice as an option.

``run_chain(weather, system, options)`` is the only implementation of
weather -> AC power. ``bench`` calls it, so every named model runs the
same chain. The thesis and current
models are named in :mod:`solarmodel.models`.

Presets
-------
``ChainOptions.frozen()``    the reference chain, bit for bit
                             (Reindl, beam-only Fresnel IAM, SAPM spectral at
                             sea-level pressure, NREL snow on hourly snowfall,
                             PVsyst thermal, De Soto, inverter-referenced losses
                             after the inverter, soiling after the AC clip).
``ChainOptions.improved()``  the review's fixes: site pressure, terrain horizon,
                             row shading and sky view when a layout is given,
                             IAM on diffuse light, snow with ground depth,
                             rolling trigger and snow albedo, DC losses before
                             the inverter with a separate AC rating (ILR),
                             soiling applied optically. Transposition stays
                             Reindl until it is checked against measured POA; row losses
                             are computed with infinite sheds (Hay-Davies) and
                             applied to it as ratios (``rows_method="ratio"``).
``ChainOptions.pvgis_like()`` PVGIS components: Muneer sky diffuse, Martin-Ruiz
                             angular losses on beam and diffuse, Faiman thermal
                             with PVGIS c-Si coefficients, Huld power model.
                             For benchmarking against PVGIS, not for sizing.

Timestamps
----------
Row labels are hour labels. Irradiance from Open-Meteo/ERA5 is the mean over
the hour *ending* at the label. Solar position is evaluated at the
representative instant: the label minus half the interval for ``"end"``
labels, the label itself for ``"center"`` (sub-hourly steps, see ``bench.simulate``).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd
import pvlib

from solarmodel.model import transposition as tr
from solarmodel.model.power import (
    spectral_correction, fit_module_params, cec_module_params, apply_loss_stack,
)
from solarmodel.losses.soiling_loss import SoilingSpec
from solarmodel.losses.snow import snow_transmittance_nrel, albedo_with_snow

_LABEL_OFFSET = {"end": -0.5, "center": 0.0}

_CHOICES = dict(
    timestamp_label=set(_LABEL_OFFSET),
    transposition={"reindl", "perez", "perez-driesse", "haydavies", "isotropic", "klucher", "king", "muneer"},
    horizon_diffuse={"circumsolar", "svf"},
    rows_transposition={"haydavies", "isotropic"},
    rows_method={"ratio", "direct"},
    iam={"physical_beam", "physical", "martin_ruiz", "none"},
    spectral={"sapm", "firstsolar", "none"},
    snow={"nrel", "none"},
    snow_trigger={"hourly", "rolling"},
    thermal={"pvsyst", "faiman", "faiman_pvgis"},
    electrical={"desoto", "pvsyst", "huld"},
    loss_order={"post_inverter", "physical"},
    inverter={"pvwatts", "sandia"},
    soiling_position={"post_ac", "optical"},
)

# PVGIS c-Si Faiman coefficients, Huld and Gracia Amillo (2015), Energies 8, 5159, Eq. 3
FAIMAN_PVGIS_CSI = (26.9, 6.20)


@dataclass
class ChainOptions:
    """Every modelling choice of the chain. Defaults = the frozen thesis chain."""
    timestamp_label: str = "end"
    interval_minutes: int = 60
    site_pressure: bool = False
    transposition: str = "reindl"
    use_horizon: bool = False
    horizon_diffuse: str = "circumsolar"
    use_rows: bool = False
    rows_transposition: str = "haydavies"
    rows_method: str = "ratio"
    iam: str = "physical_beam"
    spectral: str = "sapm"
    snow: str = "nrel"
    snow_use_depth: bool = False
    snow_trigger: str = "hourly"
    snow_trigger_window_h: int = 6
    snow_threshold_cm: float = 1.0
    snow_albedo: Optional[float] = None
    thermal: str = "pvsyst"
    wind_height_m: Optional[float] = None
    electrical: str = "desoto"
    huld_version: str = "pvgis6"
    loss_order: str = "post_inverter"
    inverter: str = "pvwatts"             # sandia: CEC/Sandia curve from [inverter].cec (physical order)
    soiling_position: str = "post_ac"

    def __post_init__(self):
        for k, allowed in _CHOICES.items():
            v = getattr(self, k)
            if v not in allowed:
                raise ValueError(f"ChainOptions.{k}={v!r}; choose one of {sorted(allowed)}")

    def solpos_offset(self) -> float:
        """Minutes added to the label to evaluate solar position."""
        return _LABEL_OFFSET[self.timestamp_label] * self.interval_minutes

    @classmethod
    def frozen(cls, **kw) -> "ChainOptions":
        return cls(**kw)

    @classmethod
    def improved(cls, **kw) -> "ChainOptions":
        base = dict(
            site_pressure=True, use_horizon=True, horizon_diffuse="circumsolar",
            use_rows=True, iam="physical", snow="nrel", snow_use_depth=True,
            snow_trigger="rolling", snow_trigger_window_h=6, snow_albedo=0.6,
            loss_order="physical", soiling_position="optical",
        )
        base.update(kw)
        return cls(**base)

    @classmethod
    def pvgis_like(cls, **kw) -> "ChainOptions":
        base = dict(
            site_pressure=True, transposition="muneer", use_horizon=True,
            iam="martin_ruiz", spectral="none", thermal="faiman_pvgis", electrical="huld",
            loss_order="physical", snow="none",
        )
        base.update(kw)
        return cls(**base)


@dataclass
class Layout:
    """Row geometry of a large array (for ``use_rows``)."""
    gcr: float = 0.40
    height_m: float = 1.5
    collector_width_m: float = 2.0


@dataclass
class PVSystem:
    """Site, array, module and balance-of-system description."""
    lat: float
    lon: float
    alt: float
    tilt: float
    azimuth: float
    albedo: float
    module: dict
    ns_arr: int
    np_arr: int
    pdc0_w: Optional[float] = None          # array STC [W]; default Imp*Vmp*Ns*Np
    iam_n: float = 1.526
    iam_k: float = 4.0
    iam_l: float = 0.002
    iam_ar: float = 0.16                     # Martin-Ruiz angular losses coefficient
    spectral_module_type: str = "monosi"
    thermal: dict = field(default_factory=lambda: dict(u_c=29.0, u_v=0.0, eta_module=0.1, alpha_abs=0.9))
    mismatch: float = 0.98
    dc_wiring: float = 0.98
    ac_wiring: float = 0.99
    inverter_eta: float = 0.97
    dc_other: float = 1.0                    # degradation, LID, nameplate tolerance ([losses] n_degradation); both loss orders
    ac_other: float = 1.0                    # transformer, availability (physical order only)
    ilr: Optional[float] = None              # DC/AC ratio; None = AC rating eta*pdc0 (thesis)
    inverter_params: Optional[dict] = None   # Sandia/CEC coefficients ([inverter].cec), for inverter="sandia"
    n_inverters: int = 1                     # identical inverters sharing the array ([inverter].count)
    export_limit_kw: float = float("inf")
    bifaciality: float = 0.0
    bifacial_shade_factor: float = -0.02
    bifacial_transmission: float = 0.0
    layout: Optional[Layout] = None
    horizon: Optional[pd.Series] = None
    tracker: Optional[dict] = None
    soiling: SoilingSpec = field(default_factory=SoilingSpec)
    snow_tilt: Optional[float] = None        # tilt for snow sliding only; None = tilt (see run_chain)

    @property
    def pdc0(self) -> float:
        if self.pdc0_w is not None:
            return float(self.pdc0_w)
        m = self.module
        return float(m["imp_ref"] * m["vmp_ref"] * self.ns_arr * self.np_arr)

    @property
    def n_modules(self) -> int:
        return int(self.ns_arr * self.np_arr)

    @classmethod
    def from_site_toml(cls, site: dict) -> "PVSystem":
        """From a parsed ``config/site/<name>.toml``. The file is checked first by
        :func:`solarmodel.schema.validate_site`: a missing required entry, an unknown
        section or key, a value out of range or an unknown provenance class raises
        :class:`solarmodel.schema.SiteConfigError` naming every offending entry."""
        from solarmodel.schema import validate_site
        validate_site(site)
        loc, arr, sysd, los = site["location"], site["array"], site["system"], site["losses"]
        iam, pvs = site["iam"], site["thermal"]["pvsyst"]
        module = dict(site["module"])
        module["ns_mod"] = module["ns"]
        opt = {}                                 # optional features: absent means "not modelled"
        if "a_r" in iam:
            opt["iam_ar"] = iam["a_r"]
        if "ac_cap_kw" in sysd:
            opt["export_limit_kw"] = float(sysd["ac_cap_kw"])
        if "pdc0_w" in sysd:
            opt["pdc0_w"] = float(sysd["pdc0_w"])
        if "n_degradation" in los:                   # age degradation, a DC loss in both loss orders
            opt["dc_other"] = float(los["n_degradation"])
        if "layout" in site:
            opt["layout"] = Layout(**site["layout"])
        if arr.get("tracking", False):
            opt["tracker"] = site["tracker"]
        if "snow_tilt" in arr:
            opt["snow_tilt"] = float(arr["snow_tilt"])
        if "bifaciality" in module:
            opt["bifaciality"] = float(module["bifaciality"])
        if "spectral" in module:                     # spectral-model module type, e.g. "monosi", "multisi"
            opt["spectral_module_type"] = module["spectral"]
        inv = site.get("inverter", {})
        if "cec" in inv:
            opt["inverter_params"] = cec_inverter(inv["cec"])
            opt["n_inverters"] = int(inv.get("count", 1))
        fai = site["thermal"].get("faiman")
        return cls(
            lat=loc["lat"], lon=loc["lon"], alt=loc["altitude"],
            tilt=float(arr["tilt"]), azimuth=float(arr["azimuth"]), albedo=float(arr["albedo"]),
            module=module, ns_arr=int(sysd["ns_arr"]), np_arr=int(sysd["np_arr"]),
            iam_n=iam["n"], iam_k=iam["k"], iam_l=iam["l"],
            thermal=dict(u_c=pvs["u_c"], u_v=pvs["u_v"], eta_module=pvs["eta_module"],
                         alpha_abs=pvs["alpha_abs"], **({"u0": fai["u0"], "u1": fai["u1"]} if fai else {})),
            mismatch=los["n_mismatch"], dc_wiring=los["n_dc_wiring"], ac_wiring=los["n_ac_wiring"],
            inverter_eta=los["n_inverter"], ilr=site.get("inverter", {}).get("ilr"),
            soiling=SoilingSpec.from_dict(site.get("soiling")),
            **opt,
        )


def cec_inverter(name: str) -> dict:
    """Sandia inverter coefficients of one entry of pvlib's CEC inverter database."""
    global _CEC_DB
    if _CEC_DB is None:
        _CEC_DB = pvlib.pvsystem.retrieve_sam("cecinverter")
    if name not in _CEC_DB.columns:
        raise KeyError(f"inverter {name!r} is not in the CEC database")
    return {k: float(v) for k, v in _CEC_DB[name].items()
            if k in ("Paco", "Pdco", "Vdco", "Pso", "C0", "C1", "C2", "C3", "Pnt",
                     "Vdcmax", "Idcmax", "Mppt_low", "Mppt_high")}


_CEC_DB = None


def solar_geometry(labels: pd.DatetimeIndex, system: PVSystem, options: ChainOptions, pressure=None) -> dict:
    """Solar position, airmass, extraterrestrial DNI and array orientation."""
    sp_times = labels + pd.Timedelta(minutes=options.solpos_offset())
    solpos = pvlib.solarposition.get_solarposition(sp_times, system.lat, system.lon, system.alt)
    solpos.index = labels
    zen, azi = solpos["apparent_zenith"], solpos["azimuth"]
    airmass = pvlib.atmosphere.get_relative_airmass(zen)
    if options.site_pressure:
        p = pressure if pressure is not None else pvlib.atmosphere.alt2pres(system.alt)
        am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, p)
    else:
        am_abs = pvlib.atmosphere.get_absolute_airmass(airmass)
    dni_extra = pvlib.irradiance.get_extra_radiation(labels)
    if system.tracker:
        t = system.tracker
        tro = pvlib.tracking.singleaxis(
            apparent_zenith=zen, solar_azimuth=azi, axis_tilt=t["axis_tilt"],
            axis_azimuth=t["axis_azimuth"], max_angle=t["max_angle"],
            backtrack=t["backtrack"], gcr=t["gcr"], cross_axis_tilt=t["cross_axis_tilt"])
        tilt, az = tro["surface_tilt"].fillna(0.0), tro["surface_azimuth"].fillna(180.0)
        aoi = tro["aoi"].fillna(90.0)
    else:
        tilt, az = system.tilt, system.azimuth
        aoi = pvlib.irradiance.aoi(tilt, az, zen, azi)
    return dict(zenith=zen, azimuth=azi, airmass=airmass, am_abs=am_abs, dni_extra=dni_extra,
                surface_tilt=tilt, surface_azimuth=az, aoi=aoi)


def plane_of_array(geo: dict, weather: pd.DataFrame, system: PVSystem, options: ChainOptions,
                   albedo=None) -> pd.DataFrame:
    """POA components: open field (optionally behind the horizon) or rows."""
    alb = system.albedo if albedo is None else albedo
    horizon = system.horizon if options.use_horizon else None
    args = (geo["surface_tilt"], geo["surface_azimuth"], geo["zenith"], geo["azimuth"],
            weather["dni"], weather["ghi"], weather["dhi"], geo["dni_extra"])
    if not (options.use_rows and system.layout is not None):
        return tr.poa_open_field(*args, geo["airmass"], alb, model=options.transposition,
                                 horizon=horizon, horizon_diffuse=options.horizon_diffuse)
    lay = system.layout
    geom = dict(gcr=lay.gcr, height=lay.height_m, collector_width=lay.collector_width_m)
    if options.rows_method == "direct":
        return tr.poa_rows(*args, alb, **geom, model=options.rows_transposition,
                           horizon=horizon, bifacial=system.bifaciality > 0)
    # "ratio": row losses from infinite sheds on an isotropic basis (beam shading,
    # sky view, ground view), applied to the chosen sky model; circumsolar light is
    # shaded like beam. Keeps the validated transposition and adds only row effects.
    rows = tr.poa_rows(*args, alb, **geom, model="isotropic", horizon=horizon, bifacial=False)
    ref = tr.poa_components(*args, geo["airmass"], alb, model="isotropic", horizon=horizon)
    comp = tr.poa_components(*args, geo["airmass"], alb, model=options.transposition,
                             horizon=horizon, horizon_diffuse=options.horizon_diffuse)

    def ratio(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(np.asarray(b, float) > 1e-6, np.asarray(a, float) / np.asarray(b, float), 1.0)
        return np.clip(r, 0.0, 1.0)

    f_beam = ratio(rows["poa_direct"], ref["direct"])
    f_sky = ratio(rows["poa_sky_diffuse"], ref["rest"])
    f_gnd = ratio(rows["poa_ground_diffuse"], ref["ground"])
    out = pd.DataFrame(index=weather.index)
    out["poa_direct"] = comp["direct"] * f_beam
    out["poa_sky_diffuse"] = comp["circumsolar"] * f_beam + comp["rest"] * f_sky
    out["poa_ground_diffuse"] = comp["ground"] * f_gnd
    out["poa_diffuse"] = out["poa_sky_diffuse"] + out["poa_ground_diffuse"]
    out["poa_global"] = out["poa_direct"] + out["poa_diffuse"]
    out["shaded_fraction"] = rows["shaded_fraction"].to_numpy()
    if system.bifaciality > 0:
        out["poa_back"] = tr.poa_rows(*args, alb, **geom, model=options.rows_transposition,
                                      horizon=horizon, bifacial=True)["poa_back"].to_numpy()
    return out


def _diffuse_iam(model: str, surface_tilt, system: PVSystem):
    """(sky, ground) diffuse IAM factors, cached per rounded tilt for trackers."""
    def one(t):
        if model == "physical":
            d = pvlib.iam.marion_diffuse("physical", t, n=system.iam_n, K=system.iam_k, L=system.iam_l)
            return float(d["sky"]), float(d["ground"])
        d = pvlib.iam.martin_ruiz_diffuse(t, a_r=system.iam_ar)
        sky, gnd = (d["sky"], d["ground"]) if isinstance(d, dict) else d   # dict from pvlib 0.16, tuple before
        return float(sky), float(gnd)
    if np.ndim(surface_tilt) == 0:
        return one(float(surface_tilt))
    t = np.round(np.asarray(surface_tilt, float), 0)
    lut = {u: one(u) for u in np.unique(t)}
    sky = np.array([lut[v][0] for v in t]); gnd = np.array([lut[v][1] for v in t])
    return sky, gnd


def effective_irradiance(poa: pd.DataFrame, geo: dict, weather: pd.DataFrame, system: PVSystem,
                         options: ChainOptions):
    """IAM- and spectrum-corrected front irradiance plus the bifacial rear gain [W/m2]."""
    aoi = geo["aoi"]
    if options.iam == "none":
        eff = (poa["poa_direct"] + poa["poa_sky_diffuse"] + poa["poa_ground_diffuse"]).clip(lower=0)
    elif options.iam == "physical_beam":   # frozen: Fresnel on beam only
        iam_b = pvlib.iam.physical(aoi, n=system.iam_n, K=system.iam_k, L=system.iam_l)
        eff = (poa["poa_direct"] * iam_b + poa["poa_sky_diffuse"] + poa["poa_ground_diffuse"]).clip(lower=0)
    else:
        if options.iam == "physical":
            iam_b = pvlib.iam.physical(aoi, n=system.iam_n, K=system.iam_k, L=system.iam_l)
        else:
            iam_b = pvlib.iam.martin_ruiz(aoi, a_r=system.iam_ar)
        f_sky, f_gnd = _diffuse_iam(options.iam, geo["surface_tilt"], system)
        eff = (poa["poa_direct"] * iam_b + poa["poa_sky_diffuse"] * f_sky
               + poa["poa_ground_diffuse"] * f_gnd).clip(lower=0)

    if options.spectral == "sapm":
        M = spectral_correction(geo["am_abs"], module_type=system.spectral_module_type)
    elif options.spectral == "firstsolar":
        if "relative_humidity" not in weather:
            raise ValueError("spectral='firstsolar' needs weather['relative_humidity']")
        pw = pvlib.atmosphere.gueymard94_pw(weather["temp_air"], weather["relative_humidity"])
        M = pvlib.spectrum.spectral_factor_firstsolar(pw, geo["am_abs"],
                                                     module_type=system.spectral_module_type)
        M = M.fillna(1.0) if hasattr(M, "fillna") else np.nan_to_num(M, nan=1.0)
    else:
        M = 1.0
    eff = (eff * M).clip(lower=0)
    back = 0.0
    if "poa_back" in poa and system.bifaciality > 0:
        back = poa["poa_back"] * system.bifaciality * (1 + system.bifacial_shade_factor) * \
            (1 + system.bifacial_transmission)
    return eff, back, M


def wind_at_module(wind, options: ChainOptions):
    """10 m wind at ``options.wind_height_m`` by the 1/7 power law; unchanged when that is None."""
    if options.wind_height_m is None:
        return wind
    return pvlib.atmosphere.windspeed_powerlaw(wind, 10.0, options.wind_height_m, exponent=1 / 7)


def cell_temperature(poa_driver, weather: pd.DataFrame, system: PVSystem, options: ChainOptions):
    """Cell (or module) temperature [degC]."""
    wind = wind_at_module(weather["wind_speed"], options)
    th = system.thermal
    if options.thermal == "pvsyst":
        return pvlib.temperature.pvsyst_cell(poa_driver, weather["temp_air"], wind,
                                             u_c=th["u_c"], u_v=th["u_v"],
                                             module_efficiency=th["eta_module"],
                                             alpha_absorption=th["alpha_abs"])
    u0, u1 = (th["u0"], th["u1"]) if options.thermal == "faiman" else FAIMAN_PVGIS_CSI
    return pvlib.temperature.faiman(poa_driver, weather["temp_air"], wind, u0=u0, u1=u1)


def module_parameters(system: PVSystem) -> dict:
    """De Soto parameters fitted from the datasheet values (thesis method), or, with
    ``[module].cec``, the CEC entry's STC curve with the datasheet alpha/beta imposed."""
    m = system.module
    if "cec" in m:
        return cec_module_params(m["cec"], m["mu_isc"], m["mu_voc"])
    kw = {}
    if "eg_ref" in m:
        kw["EgRef"] = m["eg_ref"]
    return fit_module_params(Isc_ref=m["isc_ref"], Voc_ref=m["voc_ref"], Imp_ref=m["imp_ref"],
                             Vmp_ref=m["vmp_ref"], mu_Isc=m["mu_isc"], mu_Voc=m["mu_voc"],
                             Ns=m["ns_mod"], **kw)


def dc_power_v(eff, t_cell, system: PVSystem, options: ChainOptions, module_params=None):
    """Array DC power [W] and string MPP voltage [V] (None for Huld, which has no voltage)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if options.electrical == "desoto":
            mp = module_params or module_parameters(system)
            p = pvlib.pvsystem.calcparams_desoto(
                effective_irradiance=eff, temp_cell=t_cell, alpha_sc=mp["alpha_sc"], a_ref=mp["a_ref"],
                I_L_ref=mp["I_L_ref"], I_o_ref=mp["I_o_ref"], R_sh_ref=mp["R_sh_ref"], R_s=mp["R_s"],
                EgRef=mp["EgRef"], dEgdT=mp["dEgdT"])
            sd = pvlib.pvsystem.singlediode(*p)
            return sd["p_mp"] * system.ns_arr * system.np_arr, sd["v_mp"] * system.ns_arr
        if options.electrical == "pvsyst":
            ps = system.module.get("pvsyst")
            if not ps:
                raise ValueError("electrical='pvsyst' needs module['pvsyst'] (PAN-file parameters)")
            p = pvlib.pvsystem.calcparams_pvsyst(eff, t_cell, **ps)
            sd = pvlib.pvsystem.singlediode(*p)
            return sd["p_mp"] * system.ns_arr * system.np_arr, sd["v_mp"] * system.ns_arr
        pdc0_mod = system.module["imp_ref"] * system.module["vmp_ref"]
        p = pvlib.pvarray.huld(eff, t_cell, pdc0_mod, cell_type=system.module.get("huld_cell_type", "cSi"),
                               k_version=options.huld_version)
        p = p.clip(lower=0) if hasattr(p, "clip") else np.clip(p, 0, None)
        return p * system.n_modules, None


def _sandia_unclipped(v_dc, p_dc, c: dict):
    """Sandia inverter AC power before the Paco and Pso limits (King et al. 2007, Eq. 1-4)."""
    a = c["Pdco"] * (1 + c["C1"] * (v_dc - c["Vdco"]))
    b = c["Pso"] * (1 + c["C2"] * (v_dc - c["Vdco"]))
    k = c["C0"] * (1 + c["C3"] * (v_dc - c["Vdco"]))
    return (c["Paco"] / (a - b) - k * (a - b)) * (p_dc - b) + k * (p_dc - b) ** 2


def ac_power_kw(pdc_w, system: PVSystem, options: ChainOptions, v_dc=None):
    """AC power after inverter and wiring losses [kW], plus clipped power [kW].

    ``post_inverter`` (frozen): PVWatts inverter referenced to the array
    nameplate, then mismatch, DC and AC wiring multiplied on the AC side.
    ``physical``: mismatch, DC wiring and ``dc_other`` before the inverter (``post_inverter``
    applies ``dc_other`` with the other factors after it); the
    inverter AC rating is ``pdc0 / ilr`` (or ``eta * pdc0`` when ``ilr`` is
    None); AC wiring and ``ac_other`` after it; then the export limit.
    ``inverter="sandia"`` (physical order only): the Sandia curve of
    ``system.inverter_params`` at the string voltage ``v_dc``, per inverter;
    the AC rating is ``Paco * n_inverters`` and ``ilr``/``inverter_eta`` are unused.
    """
    if options.inverter == "sandia":
        if options.loss_order != "physical":
            raise ValueError("inverter='sandia' needs loss_order='physical' (DC losses before the inverter)")
        c = system.inverter_params
        if c is None:
            raise ValueError("inverter='sandia' needs [inverter].cec in the site file")
        if v_dc is None:
            raise ValueError("inverter='sandia' needs a string voltage (electrical='desoto' or 'pvsyst')")
        n = system.n_inverters
        pdc_l = np.asarray(pdc_w * system.mismatch * system.dc_wiring * system.dc_other, float) / n
        v = np.asarray(v_dc, float) * system.dc_wiring          # resistive drop: same fraction as the power loss
        with np.errstate(all="ignore"):
            unclipped = np.where(pdc_l > c["Pso"], _sandia_unclipped(v, pdc_l, c), 0.0)
        unclipped = np.nan_to_num(np.clip(unclipped, 0.0, None))
        pac = np.minimum(unclipped, c["Paco"])                 # night tare (Pnt) not drawn from the meter
        idx = pdc_w.index if isinstance(pdc_w, pd.Series) else None
        p = pd.Series(pac * n / 1000.0, index=idx) * system.ac_wiring * system.ac_other
        p_lim = p.clip(lower=0, upper=system.export_limit_kw)
        clipped = (unclipped - pac) * n / 1000.0 * system.ac_wiring * system.ac_other + (p - p_lim).to_numpy()
        return p_lim, pd.Series(clipped, index=idx)
    if options.loss_order == "post_inverter":
        pac_w = pvlib.inverter.pvwatts(pdc=pdc_w, pdc0=system.pdc0, eta_inv_nom=system.inverter_eta)
        p = apply_loss_stack(pac_w / 1000.0, mismatch=system.mismatch,
                             wiring_dc=system.dc_wiring, wiring_ac=system.ac_wiring) * system.dc_other
        return p.clip(lower=0, upper=system.export_limit_kw), 0.0 * p
    pdc_l = pdc_w * system.mismatch * system.dc_wiring * system.dc_other
    pac0 = system.pdc0 / system.ilr if system.ilr else system.inverter_eta * system.pdc0
    pdc0_inv = pac0 / system.inverter_eta
    pac_w = pvlib.inverter.pvwatts(pdc=pdc_l, pdc0=pdc0_inv, eta_inv_nom=system.inverter_eta)
    with np.errstate(all="ignore"):
        z = np.asarray(pdc_l, float) / pdc0_inv
        eta = system.inverter_eta / 0.9637 * (-0.0162 * z - 0.0059 / z + 0.9858)
        unclipped = np.where(z > 0, eta * np.asarray(pdc_l, float), 0.0)
    clip_w = np.clip(unclipped - np.asarray(pac_w, float), 0.0, None)
    p = (pac_w / 1000.0) * system.ac_wiring * system.ac_other
    p_lim = p.clip(lower=0, upper=system.export_limit_kw)
    clipped = clip_w / 1000.0 * system.ac_wiring * system.ac_other + np.asarray(p - p_lim, float)
    return p_lim, pd.Series(clipped, index=p.index) if isinstance(p, pd.Series) else clipped


def run_chain(weather: pd.DataFrame, system: PVSystem, options: ChainOptions, *,
              soiling_ratio: pd.Series | None = None, snow_initial_coverage: float = 0.0,
              module_params: dict | None = None) -> pd.DataFrame:
    """Weather -> hourly PV output.

    weather : DataFrame on a tz-aware hourly index (row labels) with ``ghi``,
        ``dhi``, ``dni`` [W/m2], ``temp_air`` [C], ``wind_speed`` [m/s, 10 m];
        optional ``relative_humidity`` [%], ``pressure`` [Pa],
        ``snowfall_cm``, ``snow_depth_cm``.
    soiling_ratio : hourly soiling ratio on the index (default 1).

    Returns a DataFrame with ``p_ac_kw`` (output), ``p_dc_kw`` (array DC
    before losses), ``clip_kw``, ``poa_global``, ``poa_eff``, ``t_cell``,
    ``snow_tx``, ``soiling``.
    """
    times = weather.index
    pressure = weather["pressure"] if (options.site_pressure and "pressure" in weather) else None
    geo = solar_geometry(times, system, options, pressure=pressure)

    albedo = None
    if options.snow_albedo is not None and "snow_depth_cm" in weather:
        albedo = albedo_with_snow(system.albedo, weather["snow_depth_cm"], options.snow_albedo)
    poa = plane_of_array(geo, weather, system, options, albedo=albedo)
    poa_gl = poa["poa_global"]
    eff, back, _M = effective_irradiance(poa, geo, weather, system, options)

    if options.snow == "nrel":
        depth = weather["snow_depth_cm"] if options.snow_use_depth else None
        if system.tracker:
            tilt_snow = geo["surface_tilt"]
        else:
            tilt_snow = system.tilt if system.snow_tilt is None else system.snow_tilt
            if tilt_snow <= 0.0:
                # coverage_nrel clears snow only by sliding (~ sin tilt): at tilt 0 the
                # modules stay covered after the first snowfall until the series ends
                raise ValueError("snow model at tilt 0 never clears the modules; set [array] snow_tilt")
        snow_tx = snow_transmittance_nrel(
            times, weather["snowfall_cm"], poa_gl, weather["temp_air"], tilt_snow, snow_depth_cm=depth,
            initial_coverage=snow_initial_coverage, threshold_snowfall=options.snow_threshold_cm,
            trigger=options.snow_trigger, window_h=options.snow_trigger_window_h)
    else:
        snow_tx = pd.Series(1.0, index=times)

    soil = soiling_ratio if soiling_ratio is not None else pd.Series(1.0, index=times)
    poa_gl_snow = (poa_gl * snow_tx).clip(lower=0)
    eff_snow = (eff * snow_tx).clip(lower=0)
    if options.soiling_position == "optical":
        eff_snow = eff_snow * soil
    if not isinstance(back, float):
        eff_snow = eff_snow + back * snow_tx
        driver = poa_gl_snow + back * snow_tx
    else:
        driver = poa_gl_snow

    t_cell = cell_temperature(driver, weather, system, options)
    pdc_w, v_dc = dc_power_v(eff_snow, t_cell, system, options, module_params=module_params)
    p_ac, clipped = ac_power_kw(pdc_w, system, options, v_dc=v_dc)
    if options.loss_order == "post_inverter":
        p_ac = p_ac.clip(lower=0)
    if options.soiling_position == "post_ac":
        p_ac = p_ac * soil

    return pd.DataFrame({
        "p_ac_kw": p_ac, "p_dc_kw": pdc_w / 1000.0, "clip_kw": clipped,
        "poa_global": poa_gl, "poa_eff": eff_snow, "t_cell": t_cell,
        "snow_tx": snow_tx, "soiling": soil,
    }, index=times)


def module_report(system: PVSystem, options: ChainOptions | None = None,
                  irradiances=(100, 200, 400, 600, 800, 1000), datasheet_rel_eff_200: float | None = None
                  ) -> dict:
    """Low-light and temperature behaviour of the module model.

    Returns relative efficiency at each irradiance (25 C, normalised to STC),
    the implied power temperature coefficient (1000 W/m2, 25 -> 50 C) and,
    if given, the gap to the datasheet's relative efficiency at 200 W/m2.
    """
    options = options or ChainOptions.frozen()
    one = replace(system, ns_arr=1, np_arr=1, pdc0_w=None)
    g = np.array(list(irradiances) + [1000.0, 1000.0], float)
    t = np.array([25.0] * len(irradiances) + [25.0, 50.0])
    p = np.asarray(dc_power_v(pd.Series(g), pd.Series(t), one, options)[0], float)
    p_stc = p[-2]
    rel = {int(gi): float(pi / (gi / 1000.0) / p_stc) for gi, pi in zip(irradiances, p[:-2])}
    out = dict(model=options.electrical, p_stc_w=float(p_stc), rel_eff=rel,
               gamma_pmp_pct_per_k=float(100 * (p[-1] / p_stc - 1) / 25.0))
    if datasheet_rel_eff_200 is not None and 200 in rel:
        out["gap_200_vs_datasheet"] = rel[200] - datasheet_rel_eff_200
    return out
