"""The PV models, each defined once.

``thesis()``   the reference chain (``ChainOptions.frozen``): Reindl, Fresnel
               IAM on beam only, SAPM spectral at sea-level pressure, NREL snow on
               hourly snowfall, PVsyst thermal, De Soto, losses after the
               inverter, soiling after it, no clip at the AC rating. Reproduces
               the stored validation series bit for bit (tests/test_bench.py).
``homer()``    the simple rating model (HOMER / PVWatts linear) on the reference
               chain's POA: NOCT cell
               temperature, PVWatts DC at ``Ypv * f_pv * eta_inv`` with ``alpha_p``,
               clipped at the AC rating; no soiling or snow. Site keys
               ``[thermal.homer]``, ``[module] alpha_p, eta_mp_stc``, ``[losses] f_pv, eta_inv``.
``repv()``     RePV, the method of the MethodsX article: identical to ``current()``.
``rating()``   the simple rating model of the article: identical to ``homer()``.
``current()``  the reviewed chain (``ChainOptions.improved``): site pressure, PVGIS
               terrain horizon, IAM on diffuse light, NREL snow with ERA5-Land
               depth, rolling trigger and snow albedo, losses in physical order,
               optical soiling, AC clip at ``[system].ac_cap_kw``.

Both take changes as keywords, so a variant is one call::

    current(iam="none")            # current model without the diffuse IAM
    thesis(snow="none", ac_clip=True)

Every keyword except ``ac_clip`` is a :class:`ChainOptions` field; a wrong name
or value raises. ``parse("current:iam=none")`` does the same from a string
(``null`` for None, ``true``/``false`` for booleans).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from solarmodel.model.chain import ChainOptions, PVSystem


@dataclass(frozen=True)
class Model:
    name: str
    options: ChainOptions
    ac_clip: bool                 # clip AC output at [system].ac_cap_kw
    kind: str = "chain"           # "homer": linear model on the chain's POA (bench.simulate)

    def system(self, system: PVSystem) -> PVSystem:
        """The site's system as this model sees it (AC clip, terrain horizon)."""
        s = system if self.ac_clip else replace(system, export_limit_kw=float("inf"))
        if self.options.use_horizon and s.horizon is None:
            from solarmodel.weather.horizon import get_pvgis_horizon
            s = replace(s, horizon=get_pvgis_horizon(s.lat, s.lon))
        return s

    @property
    def needs_snow_depth(self) -> bool:
        o = self.options
        return (o.snow == "nrel" and o.snow_use_depth) or o.snow_albedo is not None


def _name(base: str, changes: dict) -> str:
    return base + "".join(f":{k}={v}" for k, v in changes.items())


def thesis(**changes) -> Model:
    name = _name("thesis", changes)
    ac_clip = changes.pop("ac_clip", False)
    return Model(name, ChainOptions.frozen(**changes), ac_clip)


def current(**changes) -> Model:
    name = _name("current", changes)
    ac_clip = changes.pop("ac_clip", True)
    return Model(name, ChainOptions.improved(**changes), ac_clip)


def homer(**changes) -> Model:
    """The thesis's linear model; ``changes`` go to the chain that supplies its POA."""
    name = _name("homer", changes)
    return Model(name, ChainOptions.frozen(**changes), ac_clip=True, kind="homer")


def repv(**changes) -> Model:
    """RePV as published (MethodsX): the same chain as :func:`current`, under its paper name."""
    m = current(**changes)
    return replace(m, name=_name("repv", changes))


def rating(**changes) -> Model:
    """The simple rating model of the MethodsX validation: :func:`homer` under its paper name."""
    m = homer(**changes)
    return replace(m, name=_name("rating", changes))


MODELS = {"thesis": thesis, "current": current, "homer": homer, "repv": repv, "rating": rating}


def _value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "null": None}.get(v.lower(), v)   # "none" stays a string (iam=none)


def parse(spec: str) -> Model:
    """``"current"``, ``"thesis:snow=none"``, ``"current:iam=none:ac_clip=false"``."""
    base, *parts = spec.split(":")
    if base not in MODELS:
        raise ValueError(f"model {base!r}; choose one of {sorted(MODELS)}")
    changes = {}
    for part in parts:
        k, sep, v = part.partition("=")
        if not sep:
            raise ValueError(f"{part!r} in {spec!r}: expected key=value")
        changes[k] = _value(v)
    return MODELS[base](**changes)
