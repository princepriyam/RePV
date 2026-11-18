"""Soiling ratio from daily rain: Kimber (2006) via ``pvlib.soiling.kimber``, or a fixed loss.

Kimber: soiling builds up linearly at ``loss_rate`` [fraction/day], is washed
off when daily rain exceeds ``cleaning_threshold_mm``, does not build up for
``grace_period_days`` after a wash, and is capped at ``max_soiling``.
Kimber et al. (2006), doi:10.1109/wcpec.2006.279690.

Empirical loss rates for California (Kimber 2006): Central Valley rural
0.0011/day, suburban 0.0019/day, Southern CA rural ~0, desert 0.0030/day.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pvlib.soiling


@dataclass
class SoilingSpec:
    """Soiling settings, from a site TOML ``[soiling]`` block.

    model : "kimber" (the thesis model), "fixed" (constant ``fixed_loss``) or "none".
    The Kimber defaults are the thesis values (0.11 %/day, 6 mm/day cleaning,
    14-day grace, 30 % cap). Calibrate ``loss_rate`` where you can.
    """
    model: str = "kimber"
    loss_rate: float = 0.0011
    cleaning_threshold_mm: float = 6.0
    grace_period_days: int = 14
    max_soiling: float = 0.30
    manual_cleaning_dates: list = field(default_factory=list)
    fixed_loss: float = 0.0

    def __post_init__(self):
        if self.model not in ("kimber", "fixed", "none"):
            raise ValueError(f"soiling model {self.model!r}; choose kimber, fixed or none")

    @classmethod
    def from_dict(cls, d: dict | None) -> "SoilingSpec":
        return cls(**(d or {}))          # an unknown or misspelt key raises TypeError


def soiling_ratio_daily(spec: SoilingSpec, rain_daily: pd.DataFrame) -> pd.Series:
    """Daily soiling ratio (1 = clean) on a UTC date index, from columns ``date``, ``rain_mm_day``."""
    rain_daily = rain_daily.sort_values("date")
    idx = pd.DatetimeIndex(pd.to_datetime(rain_daily["date"]))
    if spec.model == "none":
        return pd.Series(1.0, index=idx.tz_localize("UTC"))
    if spec.model == "fixed":
        return pd.Series(1.0 - spec.fixed_loss, index=idx.tz_localize("UTC"))
    loss = pvlib.soiling.kimber(
        rainfall=pd.Series(rain_daily["rain_mm_day"].to_numpy(float), index=idx),
        cleaning_threshold=spec.cleaning_threshold_mm, soiling_loss_rate=spec.loss_rate,
        grace_period=spec.grace_period_days, max_soiling=spec.max_soiling,
        manual_wash_dates=[pd.Timestamp(d).date() for d in spec.manual_cleaning_dates] or None,
        initial_soiling=0.0, rain_accum_period=24,   # hours: one daily value
    )
    return pd.Series(1.0 - loss.to_numpy().clip(0.0, spec.max_soiling), index=idx.tz_localize("UTC"))
