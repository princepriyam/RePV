"""Module fit, spectral factor and the loss stack used by the chain."""
from __future__ import annotations

import numpy as np
import pvlib

_k_over_q = 8.617333262e-5   # k/q [V/K]


def fit_module_params(Isc_ref, Voc_ref, Imp_ref, Vmp_ref,
                      mu_Isc, mu_Voc, Ns,
                      EgRef=1.121, dEgdT=-0.0002677, n_ideality=1.1):
    """
    Fit De Soto five parameters from datasheet values via pvlib.

    Parameters
    ----------
    Isc_ref    : short-circuit current at STC [A]
    Voc_ref    : open-circuit voltage at STC [V]
    Imp_ref    : MPP current at STC [A]
    Vmp_ref    : MPP voltage at STC [V]
    mu_Isc     : temperature coefficient of Isc [A/°C]
    mu_Voc     : temperature coefficient of Voc [V/°C]
    Ns         : cells in series
    EgRef      : bandgap at 0 K [eV]  (Si default 1.121)
    dEgdT      : temperature coefficient of bandgap [1/K]
    n_ideality : diode ideality factor (used for initial guess only)

    Returns
    -------
    dict with keys: alpha_sc, a_ref, I_L_ref, I_o_ref, R_s, R_sh_ref, EgRef, dEgdT
    """
    Tref = 298.15

    _n_grid   = [n_ideality, 1.0, 1.2, 1.3, 1.5]
    _rsh_mult = [1.0, 2.0, 5.0, 10.0, 0.5]

    last_err = None
    for n in _n_grid:
        for rm in _rsh_mult:
            a0   = n * Ns * _k_over_q * Tref
            IL0  = Isc_ref
            Io0  = IL0 * np.exp(-Voc_ref / a0)
            Rs0  = (Voc_ref - Vmp_ref) / Imp_ref * 0.3
            Rsh0 = max(Vmp_ref / max(Isc_ref - Imp_ref, 1e-3), 300.0) * rm
            try:
                result, _ = pvlib.ivtools.sdm.fit_desoto(
                    v_mp=Vmp_ref,
                    i_mp=Imp_ref,
                    v_oc=Voc_ref,
                    i_sc=Isc_ref,
                    alpha_sc=mu_Isc,
                    beta_voc=mu_Voc,
                    cells_in_series=Ns,
                    EgRef=EgRef,
                    dEgdT=dEgdT,
                    init_guess={'IL_0': IL0, 'Io_0': Io0,
                                'Rs_0': Rs0, 'Rsh_0': Rsh0, 'a_0': a0},
                )
                return {
                    'alpha_sc': result['alpha_sc'],
                    'a_ref':    result['a_ref'],
                    'I_L_ref':  result['I_L_ref'],
                    'I_o_ref':  result['I_o_ref'],
                    'R_s':      result['R_s'],
                    'R_sh_ref': result['R_sh_ref'],
                    'EgRef':    EgRef,
                    'dEgdT':    dEgdT,
                }
            except RuntimeError as e:
                last_err = e

    raise RuntimeError(
        f"fit_module_params: all initial-guess combinations failed.\n{last_err}"
    )


_CEC_MOD = None


def cec_module_params(name, mu_Isc, mu_Voc, dEgdT=-0.0002677):
    """
    De Soto parameters from a CEC database entry, with the datasheet temperature
    coefficients imposed: the same contract as :func:`fit_module_params` (alpha_sc
    and beta_voc honoured, gamma follows from the model).

    For modules whose datasheet point the De Soto fit cannot represent (only
    negative-shunt solutions, e.g. Suniva OPT270-60 with beta -0.353 %/K). The CEC
    entry supplies the STC curve (a_ref, I_L_ref, I_o_ref, R_s, R_sh_ref); alpha_sc
    is mu_Isc as given; EgRef is solved so that dVoc/dT at 1000 W/m2, 25 C equals
    mu_Voc. CEC's own sixth parameter (Adjust) is not used: it trades the Isc and Voc
    coefficients against gamma, so swapping a template's alpha/beta would not work.
    """
    global _CEC_MOD
    if _CEC_MOD is None:
        _CEC_MOD = pvlib.pvsystem.retrieve_sam("CECMod")
    if name not in _CEC_MOD.columns:
        raise KeyError(f"cec_module_params: {name!r} not in pvlib's CEC module database")
    c = _CEC_MOD[name]
    stc = dict(a_ref=float(c["a_ref"]), I_L_ref=float(c["I_L_ref"]), I_o_ref=float(c["I_o_ref"]),
               R_s=float(c["R_s"]), R_sh_ref=float(c["R_sh_ref"]))

    def dvoc_dt(eg):
        v = []
        for t in (20.0, 30.0):
            p = pvlib.pvsystem.calcparams_desoto(1000.0, t, mu_Isc, EgRef=eg, dEgdT=dEgdT, **stc)
            v.append(float(pvlib.pvsystem.singlediode(*p)["v_oc"]))
        return (v[1] - v[0]) / 10.0

    from scipy.optimize import brentq
    try:
        eg = brentq(lambda e: dvoc_dt(e) - mu_Voc, 0.5, 2.5, xtol=1e-6)
    except ValueError as e:
        raise RuntimeError(f"cec_module_params: no EgRef in [0.5, 2.5] gives dVoc/dT = {mu_Voc} V/K "
                           f"for {name}") from e
    return dict(alpha_sc=float(mu_Isc), EgRef=float(eg), dEgdT=dEgdT, **stc)


_SAPM_SPECTRAL_COEFF = {
    'monosi':  {'A0': 0.918093, 'A1': 0.086257, 'A2': -0.024459, 'A3': 0.002816,  'A4': -0.000126},
    'multisi': {'A0': 0.936858, 'A1': 0.069816, 'A2': -0.023798, 'A3': 0.002798,  'A4': -0.000126},
}


def spectral_correction(airmass_absolute, module_type='monosi'):
    """
    SAPM spectral mismatch correction factor (degree-4 polynomial in AM).

    Parameters
    ----------
    airmass_absolute : Series or array, absolute airmass [-]
    module_type      : 'monosi' | 'multisi'

    Returns
    -------
    Spectral correction factor M (multiplicative, typically 0.95–1.05).
    NaN entries (night-time) are filled with 1.0 since POA is zero anyway.
    """
    coeffs = _SAPM_SPECTRAL_COEFF[module_type]
    am = np.clip(airmass_absolute, 0, 10)
    M = pvlib.spectrum.spectral_factor_sapm(am, coeffs)
    if hasattr(M, 'fillna'):
        M = M.fillna(1.0)
    return M


def apply_loss_stack(P_dc, mismatch=0.98, wiring_dc=0.98, wiring_ac=0.99):
    """
    Apply sequential multiplicative loss stack to DC power.

    Parameters
    ----------
    P_dc       : Series or array, DC power [W or kW]
    mismatch   : mismatch efficiency  (default 2 % loss)
    wiring_dc  : DC wiring efficiency (default 2 % loss)
    wiring_ac  : AC wiring efficiency (default 1 % loss)

    Returns
    -------
    Power after losses, same units and type as input.
    """
    return P_dc * mismatch * wiring_dc * wiring_ac
