"""Plane-of-array irradiance: transposition models, terrain horizon, and row geometry.

Three entry points, all returning the same component frame
(``poa_direct``, ``poa_sky_diffuse``, ``poa_ground_diffuse``, ``poa_global``):

* :func:`poa_open_field`  a single, unshaded plane (the thesis assumption).
  With ``horizon=None`` this is exactly ``pvlib.irradiance.get_total_irradiance``,
  so the frozen chain is reproduced bit for bit.
* :func:`poa_rows`  rows of a large array (``pvlib.bifacial.infinite_sheds``):
  row-to-row beam shading, reduced sky view and ground view, and the rear
  side for bifacial modules.
* :func:`muneer_pvgis`  the Muneer (1990) sky-diffuse model in the form used by
  PVGIS and GRASS ``r.sun`` (Hofierka and Suri, GPL-2), selectable as
  ``model="muneer"`` in :func:`poa_open_field`.

Horizon handling follows the PVGIS/``r.sun`` logic: when the sun is below the
terrain horizon at its azimuth, beam and circumsolar diffuse are removed
(Muneer switches to its "shadowed surface" form). ``horizon_diffuse="svf"``
additionally scales the isotropic sky by the share of the sky dome the
terrain leaves visible to the tilted plane (:func:`horizon_sky_view_ratio`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def muneer_pvgis(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                 solar_zenith, solar_azimuth, shaded=None, return_components=False):
    """Sky diffuse on a tilted plane, Muneer (1990) as implemented in PVGIS/r.sun.

    Coefficients follow GRASS ``r.sun`` (``rsunlib.c::drad``): the sunlit form
    uses ``(0.00263 - 0.712 Kb - 0.6883 Kb^2)``, the shadowed form the constant
    0.252271 (Muneer's b = 5.73), with the low-sun variant below 0.1 rad.

    Parameters
    ----------
    surface_tilt, surface_azimuth : numeric  [deg], pvlib convention (N=0, E=90).
    dhi, dni, dni_extra : numeric  [W/m2]
    solar_zenith, solar_azimuth : numeric  [deg]
    shaded : bool array, optional  True where the plane is shadowed by terrain;
        the shadowed form is then used and the circumsolar term drops out.
    return_components : bool  also return ``circumsolar`` and ``isotropic``.

    Returns
    -------
    Series/array of sky diffuse [W/m2], or a DataFrame/dict with
    ``sky_diffuse``, ``circumsolar``, ``isotropic`` if ``return_components``.

    References
    ----------
    Muneer, T. (1990). Solar radiation model for Europe. Building Services
    Engineering Research and Technology 11(4), 153-163.
    GRASS GIS r.sun manual and source (raster/r.sun/rsunlib.c), GPL-2.0-or-later.
    """
    beta = np.radians(surface_tilt)
    h = np.radians(90.0 - np.asarray(solar_zenith, dtype=float))
    sin_h = np.sin(h)
    dhi_a = np.asarray(dhi, dtype=float)
    dni_a = np.asarray(dni, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kb = np.where(sin_h > 0, np.clip(dni_a / np.asarray(dni_extra, dtype=float), 0.0, 1.0), 0.0)
    cos_inc = np.clip(pvlib.irradiance.aoi_projection(surface_tilt, surface_azimuth,
                                                      solar_zenith, solar_azimuth), 0.0, None)
    cos_inc = np.asarray(cos_inc, dtype=float)
    r_sky = (1.0 + np.cos(beta)) / 2.0
    fg = np.sin(beta) - beta * np.cos(beta) - np.pi * np.sin(beta / 2.0) ** 2
    n_term = (0.00263 - 0.712 * kb - 0.6883 * kb ** 2) * fg + r_sky

    shadow = (cos_inc <= 0) | (sin_h <= 0)
    if shaded is not None:
        shadow = shadow | np.asarray(shaded, dtype=bool)
    low_sun = h < 0.1
    a_ln = np.radians(np.asarray(solar_azimuth, dtype=float) - surface_azimuth)
    with np.errstate(divide="ignore", invalid="ignore"):
        circ_hi = kb * cos_inc / np.where(sin_h > 0, sin_h, np.nan)
        circ_lo = kb * np.sin(beta) * np.cos(a_ln) / (0.1 - 0.008 * h)
    circ = np.where(low_sun, circ_lo, circ_hi)
    circ = np.where(shadow, 0.0, np.nan_to_num(circ, nan=0.0))
    iso = np.where(shadow, r_sky + 0.252271 * fg, n_term * (1.0 - kb))
    # r.sun treats a horizontal plane separately: diffuse = DHI
    flat = np.asarray(beta) == 0
    circ, iso = np.where(flat, 0.0, circ), np.where(flat, 1.0, iso)
    sky = np.clip(dhi_a * (iso + circ), 0.0, None)

    if not return_components:
        return pd.Series(sky, index=dhi.index) if isinstance(dhi, pd.Series) else sky
    out = {"sky_diffuse": sky, "circumsolar": np.clip(dhi_a * circ, 0, None),
           "isotropic": np.clip(dhi_a * iso, 0, None)}
    if isinstance(dhi, pd.Series):
        return pd.DataFrame(out, index=dhi.index)
    return out


def horizon_elevation_at(solar_azimuth, horizon: pd.Series):
    """Terrain horizon elevation [deg] at each solar azimuth (circular interpolation).

    ``horizon`` is indexed by azimuth [deg, N=0 clockwise], as returned by
    :func:`pvlib.iotools.get_pvgis_horizon`.
    """
    h = horizon.sort_index()
    az = np.asarray(h.index, dtype=float) % 360.0
    el = np.asarray(h.values, dtype=float)
    order = np.argsort(az)
    az, el = az[order], el[order]
    az_ext = np.concatenate([az - 360.0, az, az + 360.0])
    el_ext = np.concatenate([el, el, el])
    out = np.interp(np.asarray(solar_azimuth, dtype=float) % 360.0, az_ext, el_ext)
    if isinstance(solar_azimuth, pd.Series):
        return pd.Series(out, index=solar_azimuth.index)
    return out


def sun_blocked_by_horizon(apparent_zenith, solar_azimuth, horizon: pd.Series | None):
    """Boolean mask, True where the sun centre is below the terrain horizon."""
    if horizon is None:
        return np.zeros(np.shape(apparent_zenith), dtype=bool)
    elev = 90.0 - np.asarray(apparent_zenith, dtype=float)
    return elev < np.asarray(horizon_elevation_at(solar_azimuth, horizon), dtype=float)


def horizon_sky_view_ratio(horizon: pd.Series, surface_tilt: float, surface_azimuth: float,
                           n_az: int = 360, n_el: int = 45) -> float:
    """Share of the isotropic sky diffuse a tilted plane still receives behind a horizon.

    Numerically integrates ``max(cos(theta), 0) cos(e) de dphi`` over the sky
    patches hidden by terrain (elevation 0 to horizon) and divides by the
    unobstructed value ``pi (1 + cos beta) / 2``. Returns 1.0 for a flat horizon.
    """
    phi = np.radians(np.linspace(0.0, 360.0, n_az, endpoint=False) + 180.0 / n_az)
    h_el = np.radians(np.clip(horizon_elevation_at(np.degrees(phi), horizon), 0.0, 90.0))
    beta, gamma = np.radians(surface_tilt), np.radians(surface_azimuth)
    nrm = np.array([np.sin(beta) * np.sin(gamma), np.sin(beta) * np.cos(gamma), np.cos(beta)])
    dphi = 2 * np.pi / n_az
    blocked = 0.0
    for p, hmax in zip(phi, h_el):
        if hmax <= 0:
            continue
        e = (np.arange(n_el) + 0.5) * hmax / n_el
        d = np.stack([np.sin(p) * np.cos(e), np.cos(p) * np.cos(e), np.sin(e)])
        cos_t = np.clip(nrm @ d, 0.0, None)
        blocked += float(np.sum(cos_t * np.cos(e)) * (hmax / n_el) * dphi)
    total = np.pi * (1.0 + np.cos(beta)) / 2.0
    return float(max(0.0, 1.0 - blocked / total))


def _frame(direct, sky, ground, like):
    direct = np.clip(np.asarray(direct, dtype=float), 0, None)
    sky = np.clip(np.asarray(sky, dtype=float), 0, None)
    ground = np.asarray(ground, dtype=float)
    d = {"poa_global": direct + sky + ground, "poa_direct": direct,
         "poa_diffuse": sky + ground, "poa_sky_diffuse": sky, "poa_ground_diffuse": ground}
    if isinstance(like, pd.Series):
        return pd.DataFrame(d, index=like.index)
    return d


def _sky_components(model, surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
                    dni, ghi, dhi, dni_extra, airmass):
    """(circumsolar, rest) split of the sky diffuse for the horizon-aware path."""
    if model == "isotropic":
        iso = pvlib.irradiance.isotropic(surface_tilt, dhi)
        return np.zeros(np.shape(dhi)), np.asarray(iso, dtype=float)
    if model == "haydavies":
        c = pvlib.irradiance.haydavies(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                                       solar_zenith, solar_azimuth, return_components=True)
        return np.asarray(c["poa_circumsolar"], float), np.asarray(c["poa_isotropic"], float)
    if model in ("perez", "perez-driesse"):
        fn = pvlib.irradiance.perez if model == "perez" else pvlib.irradiance.perez_driesse
        c = fn(surface_tilt, surface_azimuth, dhi, dni, dni_extra, solar_zenith, solar_azimuth,
               airmass, return_components=True)
        return (np.asarray(c["poa_circumsolar"], float),
                np.asarray(c["poa_isotropic"], float) + np.asarray(c["poa_horizon"], float))
    if model == "reindl":
        # same terms as pvlib.irradiance.reindl, split into AI*Rb and the rest
        cos_tt = np.maximum(pvlib.irradiance.aoi_projection(surface_tilt, surface_azimuth,
                                                            solar_zenith, solar_azimuth), 0)
        cos_z = np.cos(np.radians(solar_zenith))
        rb = cos_tt / np.maximum(cos_z, 0.01745)
        ai = np.asarray(dni, float) / np.asarray(dni_extra, float)
        hb = np.maximum(np.asarray(dni, float) * cos_z, 0)
        ghi_a = np.asarray(ghi, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            hb_ghi = np.where(ghi_a == 0, 0, np.divide(hb, ghi_a))
        rest = (1 - ai) * 0.5 * (1 + np.cos(np.radians(surface_tilt))) * \
            (1 + np.sqrt(hb_ghi) * np.sin(np.radians(0.5 * surface_tilt)) ** 3)
        dhi_a = np.asarray(dhi, float)
        return np.asarray(dhi_a * ai * rb, float), np.asarray(dhi_a * rest, float)
    raise ValueError(f"horizon handling not implemented for transposition model {model!r}")


def poa_components(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
                   dni, ghi, dhi, dni_extra, airmass, albedo, model="reindl",
                   horizon: pd.Series | None = None, horizon_diffuse: str = "circumsolar") -> dict:
    """Beam, circumsolar, remaining sky diffuse and ground-reflected on the plane
    [W/m2], with the horizon applied. Models: reindl, perez, perez-driesse,
    haydavies, isotropic, muneer."""
    blocked = sun_blocked_by_horizon(solar_zenith, solar_azimuth, horizon)
    if model == "muneer":
        sky = muneer_pvgis(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                           solar_zenith, solar_azimuth, shaded=blocked, return_components=True)
        rest, circ = np.asarray(sky["isotropic"]), np.asarray(sky["circumsolar"])
    else:
        circ, rest = _sky_components(model, surface_tilt, surface_azimuth, solar_zenith,
                                     solar_azimuth, dni, ghi, dhi, dni_extra, airmass)
        circ = np.where(blocked, 0.0, circ)
    if horizon is not None and horizon_diffuse == "svf":
        rest = rest * horizon_sky_view_ratio(horizon, float(np.mean(surface_tilt)), float(surface_azimuth))
    cos_inc = np.clip(pvlib.irradiance.aoi_projection(surface_tilt, surface_azimuth,
                                                      solar_zenith, solar_azimuth), 0, None)
    direct = np.where(blocked, 0.0, np.clip(np.asarray(dni, float) * np.asarray(cos_inc, float), 0, None))
    ground = np.asarray(pvlib.irradiance.get_ground_diffuse(surface_tilt, ghi, albedo), float)
    return dict(direct=direct, circumsolar=np.clip(circ, 0, None), rest=np.clip(rest, 0, None), ground=ground)


def poa_open_field(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
                   dni, ghi, dhi, dni_extra, airmass, albedo, model="reindl",
                   horizon: pd.Series | None = None, horizon_diffuse: str = "circumsolar"):
    """Unshaded plane-of-array irradiance, optionally behind a terrain horizon.

    ``model``: any pvlib ``get_total_irradiance`` model, or ``"muneer"`` (PVGIS).
    ``horizon``: Series of horizon elevation [deg] indexed by azimuth, or None.
    ``horizon_diffuse``: ``"circumsolar"`` (PVGIS logic: remove circumsolar when
    the sun is hidden) or ``"svf"`` (also scale the remaining sky diffuse by
    :func:`horizon_sky_view_ratio`; fixed tilt only).
    """
    if horizon is None and model != "muneer":
        # identical call to the frozen thesis chain
        return pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
            solar_zenith=solar_zenith, solar_azimuth=solar_azimuth,
            dni=dni, ghi=ghi, dhi=dhi, dni_extra=dni_extra, airmass=airmass,
            albedo=albedo, model=model)
    c = poa_components(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, ghi, dhi,
                       dni_extra, airmass, albedo, model=model, horizon=horizon,
                       horizon_diffuse=horizon_diffuse)
    return _frame(c["direct"], c["circumsolar"] + c["rest"], c["ground"], dhi)


def poa_rows(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, ghi, dhi,
             dni_extra, albedo, *, gcr, height, collector_width, model="haydavies",
             horizon: pd.Series | None = None, bifacial: bool = False):
    """Front (and rear) irradiance on rows of an array, ``pvlib.bifacial.infinite_sheds``.

    Geometry: ``gcr`` ground coverage ratio [-], ``height`` of the row centre
    above ground [m], ``collector_width`` slant length of one row [m]; the
    pitch is ``collector_width / gcr``. ``model`` is ``"haydavies"`` or
    ``"isotropic"`` (the two infinite_sheds supports). IAM is *not* applied
    here (iam=1) so the chain applies the same IAM model as on an open field.

    Horizon: DNI is set to zero while the sun is hidden. With Hay-Davies that
    also turns the circumsolar share isotropic rather than removing it, a
    small over-estimate that only matters on sites with a real horizon.

    Returns a DataFrame with the standard front components, plus
    ``poa_back`` [W/m2] (0 unless ``bifacial``) and ``shaded_fraction``.
    """
    if model not in ("haydavies", "isotropic"):
        raise ValueError("rows support transposition 'haydavies' or 'isotropic' only")
    blocked = sun_blocked_by_horizon(solar_zenith, solar_azimuth, horizon)
    dni_in = pd.Series(np.where(blocked, 0.0, np.asarray(dni, float)), index=ghi.index) \
        if isinstance(ghi, pd.Series) else np.where(blocked, 0.0, dni)
    pitch = collector_width / gcr
    r = pvlib.bifacial.infinite_sheds.get_irradiance(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, gcr, height, pitch,
        ghi, dhi, dni_in, albedo, model=model, dni_extra=dni_extra,
        iam_front=1.0, iam_back=1.0, bifaciality=0.0)
    out = pd.DataFrame({
        "poa_global": r["poa_front"], "poa_direct": r["poa_front_direct"],
        "poa_diffuse": r["poa_front_diffuse"], "poa_sky_diffuse": r["poa_front_sky_diffuse"],
        "poa_ground_diffuse": r["poa_front_ground_diffuse"],
        "poa_back": r["poa_back"] if bifacial else 0.0 * r["poa_front"],
        "shaded_fraction": r["shaded_fraction_front"],
    })
    return out
