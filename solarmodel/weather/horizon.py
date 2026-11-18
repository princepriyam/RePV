"""Terrain horizon profiles: PVGIS (DEM-derived) with a local cache.

PVGIS computes the profile from a ~90 m (3 arc-second) elevation model in 48
directions. It does not include near obstructions (trees, buildings); add
those with :func:`merge_horizons` from a site survey or a fisheye photo
digitised to (azimuth, elevation) pairs.

All profiles here use the pvlib convention: index = azimuth [deg] clockwise
from north (south = 180), values = horizon elevation [deg].
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from solarmodel.weather import cache
from solarmodel.weather.utils import cache_file


def get_pvgis_horizon(lat: float, lon: float, refresh: bool = False) -> pd.Series:
    """PVGIS horizon profile for (lat, lon), cached under ``cache/horizon``."""
    fp = cache_file("horizon", "pvgis_horizon", lat, lon)
    hit = None if refresh else cache.read(fp)    # parquet; a legacy CSV is converted and deleted
    if hit is not None:
        s = hit.iloc[:, 0]
        s.index = s.index.astype(float)
        return s.rename("horizon_elevation")
    import pvlib
    s, _meta = pvlib.iotools.get_pvgis_horizon(lat, lon)
    s = s.astype(float).rename("horizon_elevation")
    s.index = s.index.astype(float)
    s.index.name = "horizon_azimuth"
    return cache.write(fp, s.to_frame()).iloc[:, 0]


def flat_horizon(step: float = 7.5) -> pd.Series:
    """A zero horizon (for tests and as a neutral default)."""
    az = np.arange(0.0, 360.0, step)
    return pd.Series(np.zeros_like(az), index=az, name="horizon_elevation")


def merge_horizons(*profiles: pd.Series, step: float = 1.0) -> pd.Series:
    """Upper envelope of several horizon profiles, resampled to ``step`` degrees
    (e.g. PVGIS terrain plus surveyed near obstructions)."""
    from solarmodel.model.transposition import horizon_elevation_at
    az = np.arange(0.0, 360.0, step)
    stacked = np.vstack([horizon_elevation_at(az, p) for p in profiles])
    return pd.Series(stacked.max(axis=0), index=az, name="horizon_elevation")
