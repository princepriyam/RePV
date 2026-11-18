"""Fit PVsyst U_c / U_v against measured module temperature (PVDAQ 34 Las Vegas, 1423 Henderson).

Temperature, not power: a U-value fitted to power would trade off against the module's
temperature coefficient and other conversion errors. Nothing here uses measured AC output.

Two fits on steady, clear hours (2011-2013):
  A  on-site sensors: module_temp, ambient_temp, wind_speed, POA (sensor rescaled to
     the chain's clear-sky POA level). Physical U of the array.
  B  chain inputs: the POA, temp_air and wind the chain feeds pvsyst_cell. The U that
     makes the chain's cell temperature match the measured one: the value for the config.
Each fit with U_v = 0 and with U_v free, and with back-to-cell offsets dT = 0 and 3 K
at 1000 W/m2 (Sandia open-rack value) since the sensor is on the module back.

Raw data: the daily files of the OEDI PVDAQ data lake (public, no account), e.g.
    aws s3 cp --no-sign-request --recursive \
        s3://oedi-data-lake/pvdata/pvdaq/csv/pvdata/system_id=1423/ data/validation/NIST/pvdaq_system_1423/raw/
(system_id=34 for Las Vegas); subfolders are searched. The chain side runs RePV on the
cached ERA5 weather of each window. Henderson has no wind sensor: fit A uses U_v = 0 only.
The article's Henderson value (U_c = 30.7) is fit B with U_v = 0 and dT = 0.

    python studies/LasVegas_34/fit_thermal.py [LasVegas_34|Henderson_1423]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from solarmodel import bench, models  # noqa: E402
from solarmodel.model.chain import wind_at_module  # noqa: E402
from solarmodel.weather import inputs  # noqa: E402

SITES = {
    # windows on the chain, raw columns, raw sample step
    "LasVegas_34": dict(
        windows=[("2011-01-01", "2011-12-31"), ("2012-01-01", "2012-12-31"), ("2013-01-01", "2013-12-31")],
        raw=ROOT / "data/validation/NIST/pvdaq_system_34/raw", step="15min",
        tm=["module_temp_f__2689"], ta="ambient_temp_f__2688", ws="wind_speed_mph__2691",
        poa="poa_irradiance__2679", fahrenheit=True,
        wind_bad_from=pd.Timestamp("2013-10-01", tz="UTC")),          # sensor reads ~0 from here on
    "Henderson_1423": dict(
        windows=[("2022-01-31", "2023-01-30")],
        raw=ROOT / "data/validation/NIST/pvdaq_system_1423/raw", step="1min",
        tm=[f"module_temp_{i}__{4864 + i}" for i in range(1, 9)], ta="ambient_temp__4864", ws=None,
        poa="poa_irradiance__4861", fahrenheit=False, wind_bad_from=None),
}
POA_MIN, KC_MIN, KC_MAX, STEADY = 300.0, 0.8, 1.2, 0.10


def load_raw(cfg: dict) -> pd.DataFrame:
    """Hourly means on the chain's hour labels (UTC, label = interval end)."""
    df = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(cfg["raw"] / "**" / "system_*__date_*.csv"), recursive=True)))
    t = pd.to_datetime(df["measured_on"]).dt.round(cfg["step"])
    c = (lambda x: (x - 32) / 1.8) if cfg["fahrenheit"] else (lambda x: x)
    tm = c(df[cfg["tm"]].to_numpy(float))
    tm[(tm <= -20) | (tm > 90)] = np.nan
    d = pd.DataFrame({
        "tm": np.nanmean(tm, axis=1),
        "ta": c(df[cfg["ta"]].to_numpy(float)),
        "ws": df[cfg["ws"]].to_numpy(float) * 0.44704 if cfg["ws"] else np.nan,
        "poa": df[cfg["poa"]].to_numpy(float),
    }, index=pd.DatetimeIndex(t).tz_localize("Etc/GMT+8").tz_convert("UTC"))   # PST, start label
    d = d[~d.index.duplicated()].sort_index()
    g = d.resample("1h", label="right", closed="left")
    h = g.mean()
    n_full = pd.Timedelta("1h") // pd.Timedelta(cfg["step"])
    h["full"] = g["tm"].count() >= 0.9 * n_full
    h["poa_span"] = (g["poa"].max() - g["poa"].min()) / h["poa"]
    return h


def chain_hours(model, name: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """The chain's thermal inputs and output on its hour labels for one window."""
    site = bench.load_site(name, start, end)
    w = inputs.weather("point", "point", site.name, site.system, site.start, site.end, {}, model)
    sim = bench.simulate(site, model, w)
    labels, clear, _, _, _ = bench.align(site)
    ghi_cs = clear["ghi"].reindex(w.index)
    kc = (w["ghi"] / ghi_cs).where(ghi_cs > 10.0)                 # clear-sky index on the hour labels
    out = pd.DataFrame({"G": sim["poa_global"], "ta": w["temp_air"], "ws": wind_at_module(w["wind_speed"], model.options),
                        "t_cell": sim["t_cell"], "kc": kc})
    return out, model.system(site.system).thermal


def pvsyst(G, ta, ws, uc, uv, eta, alpha, dt):
    return ta + alpha * G * (1 - eta) / (uc + uv * ws) + dt * G / 1000.0


def fit(G, ta, ws, tm, eta, alpha, dt, free_uv: bool):
    def res(p):
        uc, uv = (p[0], p[1]) if free_uv else (p[0], 0.0)
        return pvsyst(G, ta, ws, uc, uv, eta, alpha, dt) - tm
    x0, lo, hi = ([20.0, 0.5], [3.0, 0.0], [80.0, 10.0]) if free_uv else ([20.0], [3.0], [80.0])
    r = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0)
    uc, uv = (r.x[0], r.x[1]) if free_uv else (r.x[0], 0.0)
    e = res(r.x)
    return dict(u_c=uc, u_v=uv, bias_k=float(np.mean(e)), rmse_k=float(np.sqrt(np.mean(e ** 2))),
                n=len(tm), u_at_mean_wind=uc + uv * float(np.mean(ws)))


def main(name: str = "LasVegas_34") -> None:
    cfg = SITES[name]
    OUT = Path(__file__).parent / "outputs" / "thermal_fit" / name
    OUT.mkdir(parents=True, exist_ok=True)
    model = models.repv()
    parts = [chain_hours(model, name, a, b) for a, b in cfg["windows"]]
    th = parts[0][1]
    ch = pd.concat(p[0] for p in parts)
    ch.columns = [f"c_{c}" for c in ch.columns]
    eta, alpha = th["eta_module"], th["alpha_abs"]

    # The driver is right if pvsyst() on the chain inputs reproduces the chain's t_cell.
    rep = pvsyst(ch.c_G, ch.c_ta, ch.c_ws, th["u_c"], th["u_v"], eta, alpha, 0.0) - ch.c_t_cell
    print(f"config U_c={th['u_c']} U_v={th['u_v']} eta={eta}; reproduce chain t_cell: "
          f"max |err| {np.nanmax(np.abs(rep[ch.c_G > 50])):.3f} K")

    h = load_raw(cfg).join(ch, how="inner")
    lag = {L: h.poa.corr(h.c_G.shift(L)) for L in range(-2, 3)}
    print("sensor POA vs chain POA, corr by lag [h]:", {k: round(v, 4) for k, v in lag.items()})

    sel = (h.full & (h.poa_span < STEADY) & (h.c_G > POA_MIN) & (h.poa > POA_MIN)
           & h.c_kc.between(KC_MIN, KC_MAX) & h.tm.notna() & h.ta.notna())
    s = h[sel].copy()
    k_poa = float(np.median(s.poa / s.c_G))
    print(f"steady clear hours: {len(s)}; sensor/chain POA median {k_poa:.3f}; "
          f"on-site minus chain air temp {np.mean(s.ta - s.c_ta):+.2f} K; "
          f"on-site wind {s.ws.mean():.2f} vs chain {s.c_ws.mean():.2f} m/s")
    s["poa_cal"] = s.poa / k_poa
    wind_ok = (s.index < cfg["wind_bad_from"]) if cfg["wind_bad_from"] is not None else None

    rows = []
    for dt in (0.0, 3.0):
        for free in (False, True):
            a = s[wind_ok] if (free and wind_ok is not None) else s
            if not (free and wind_ok is None):                   # no on-site wind: A with U_v = 0 only
                ws = a.ws.to_numpy() if not free else a.ws.to_numpy()
                rows.append(dict(fit="A on-site", dT_back=dt, u_v_free=free,
                                 **fit(a.poa_cal.to_numpy(), a.ta.to_numpy(), np.nan_to_num(ws), a.tm.to_numpy(),
                                       eta, alpha, dt, free)))
            rows.append(dict(fit="B chain", dT_back=dt, u_v_free=free,
                             **fit(a.c_G.to_numpy(), a.c_ta.to_numpy(), a.c_ws.to_numpy(), a.tm.to_numpy(),
                                   eta, alpha, dt, free)))
    res = pd.DataFrame(rows)
    # Chain as configured, against the same measured module temperature
    cfg_err = (s.c_t_cell - s.tm)
    print(f"chain as configured (U_c={th['u_c']}): t_cell - t_module bias {cfg_err.mean():+.2f} K, "
          f"rmse {np.sqrt((cfg_err ** 2).mean()):.2f} K")
    # residual temperature dependence for the B, U_v=0, dT=0 fit
    b = res[(res.fit == "B chain") & (res.dT_back == 0) & (~res.u_v_free)].iloc[0]
    e = pvsyst(s.c_G, s.c_ta, s.c_ws, b.u_c, 0.0, eta, alpha, 0.0) - s.tm
    print(f"B (U_v=0) residual vs measured module temp: slope {np.polyfit(s.tm, e, 1)[0]:+.3f} K/K; "
          f"by season mean err: " + ", ".join(f"{q}:{v:+.2f}" for q, v in
                                              e.groupby(s.index.quarter).mean().items()))
    print(res.round(3).to_string(index=False))
    res.to_csv(OUT / "u_fit.csv", index=False)
    s.to_csv(OUT / "hours.csv.gz")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
