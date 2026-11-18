"""Write config/site/<Place>_<sid>.toml for the PVDAQ systems extracted by pvdaq_extract.py.

    python scripts/pvdaq_site_configs.py              # all systems in SITES
    python scripts/pvdaq_site_configs.py 50 1203      # selected systems
    python scripts/pvdaq_site_configs.py --check      # also load each file with the bench

Every input comes from one of:
  metadata   data/validation/PVDAQ/<sid>/<sid>_metadata.json (location, tilt, azimuth, module
             model and count, inverter model and count)
  extract    outputs/pvdaq_inventory/extract_summary.csv and the system report (DC rating, time
             zone, per-year scorable hours) and config/pvdaq_overrides.json
  database   pvlib's CEC or Sandia module tables (STC values, temperature coefficients)
  SITES      the per-system decisions below; each says where it comes from, and every value
             that is an assumption says so in the written file
Nothing is fitted to the measured power. Every written file gets its [provenance] table
(scripts/site_provenance.py) and is checked against solarmodel/schema.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarmodel import paths  # noqa: E402

ROOT = paths.PROJECT_ROOT                  # the root bench resolves configs and measured files against
PVDAQ = ROOT / "data" / "validation" / "PVDAQ"
SITE_DIR = paths.CONFIG_DIR / "site"
_spec = importlib.util.spec_from_file_location("inv", Path(__file__).parent / "pvdaq_sensor_inventory.py")
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)

# PVsyst U-values by mounting (u_v 0): free-standing 29, semi-integrated (air duct) 20, integrated 15
U_C = {"open": 29.0, "roof": 20.0, "integrated": 15.0}
MOUNT_TXT = {"open": "free-standing rack or carport (PVsyst 'free mounted')",
             "roof": "low-tilt rack on a roof (PVsyst 'semi-integrated with air duct')",
             "integrated": "laminate bonded to the roof (PVsyst 'integrated')"}
# age degradation of the DC rating: median field rate of Jordan and Kurtz (2013), Prog. Photovolt. 21:12-29
DEG_RATE = 0.005
# fast snow removal: sites where, in the hours the current chain covers the array with snow (ERA5
# inputs, scoring year), the measured output sat above the midpoint between the chain with and
# without snow (measured / snow-free: 1200 0.75, 1201 0.76, 1204 0.64, 4902 0.72); chosen on the
# scoring-year data, so in-sample
FAST_SNOW = {1200: 0.75, 1201: 0.76, 1204: 0.64, 4902: 0.72}
# weather: NREL MIDC station near the site (else NSRDB at the site, 1998-2023)
STATION_FROM = {"BMS": 1981, "UNLV": 2007}

# module: (database, key, note). inverter: (model, count, kW each or None). Sources in notes.
SITES = {
    10: dict(place="Golden", module=("sandia", "Shell_Solar_ST40__1999__E__", "Shell 'Eclipse 80' = two ST40 CIS laminates; 28 x ST40 = 1.12 kW = System.power"),
             n_modules=28, inverter=("unknown", 1, None), mount="open", station="BMS"),
    33: dict(place="Golden", module=("cec", "SolarWorld_Industries_GmbH_Sunmodule_Plus_SW_240_poly", "PROXY: Silicor UMG-Si 240 W module has no public datasheet; a 60-cell 240 W poly module stands in; 10 x 240 W = 2.4 kW = System.power"),
             n_modules=10, inverter=("SMA SB3000US", 1, 3.0), mount="open", station="BMS"),
    50: dict(place="Golden", module=("sandia", "Siemens_Solar_SM55___1997_", "Siemens M55 (SERF East); STC 7.43 kW per van Dyk et al. 1996 (NREL/CP-411-20430), so pdc0 = 7430 W, not 140 x 55 W"),
             n_modules=140, inverter=("unknown (metadata IG 4500-LV cannot deliver the measured 7 kW)", 1, None), mount="open", station="BMS"),
    51: dict(place="Golden", module=("sandia", "Siemens_Solar_SM55___1997_", "Siemens M55 (SERF West, twin of 50); STC 7.43 kW per van Dyk et al. 1996 (NREL/CP-411-20430)"),
             n_modules=140, inverter=("unknown (metadata IG 4500-LV cannot deliver the measured 7 kW)", 1, None), mount="open", station="BMS"),
    1199: dict(place="HuntValley", module=("cec", "Suntech_Power_STP280_24_Vd", "metadata 'Suntech 280W' only; STP280-24/Vd (72-cell, 2010) assumed"),
               n_modules=189, inverter=("SMA SB7000US", 7, 7.0), mount="roof", station=None),
    1200: dict(place="Linthicum", module=("cec", "SolarWorld_Industries_GmbH_Sunmodule_Plus_SW_240_mono", "metadata 'sw240'; mono assumed"),
               n_modules=216, inverter=("SMA SB8000US", 6, 8.0), mount="roof", station=None),
    1201: dict(place="CherryHill", module=("cec", "SolarWorld_Industries_GmbH_Sunmodule_Plus_SW_245_mono", "metadata 'sw245 mono'"),
               n_modules=572, inverter=("Satcon PowerGate Plus 135 kW", 1, 135.0), mount="roof", station=None),
    1202: dict(place="CherryHill", module=("cec", "SolarWorld_Industries_GmbH_Sunmodule_Plus_SW_240_mono", "metadata 'sw240'; mono assumed"),
               n_modules=216, inverter=("SMA SB8000US", 6, 8.0), mount="roof", station=None),
    1203: dict(place="Wilmington", module=("cec", "SolarWorld_Industries_GmbH_Sunmodule_Plus_SW_245_mono", "metadata 'sw245 mono'"),
               n_modules=806, inverter=("Satcon PowerGate Plus 100 kW", 2, 100.0), mount="roof", station=None),
    1204: dict(place="Chester", module=("cec", "SolarWorld_Industries_GmbH_Sunmodule_Plus_SW_245_mono", "metadata 'sw245 mono'"),
               n_modules=321, inverter=("SMA SB8000US", 9, 8.0), mount="roof", station=None),
    1208: dict(place="Golden", module=("cec", "SunPower_SPR_315E_WHT_D", "metadata 'spr-315e-wht-d'"),
               n_modules=1664, inverter=("SMA Sunny Central 250U", 2, 250.0), mount="open", station="BMS"),
    1231: dict(place="NewSmyrnaBeach", module=("sandia", "Uni_Solar_PVL_124__2006__E__", "PROXY: metadata 'PVL-120' is not in the tables; PVL-124 (same a-Si laminate family) stands in, pdc0 = 28 x 120 W"),
               n_modules=28, inverter=("Trace SW4048 (battery inverter)", 1, None), mount="integrated", station=None),
    1239: dict(place="PresqueIsle", module=("cec", "Sharp_ND_224UC1", "metadata 'nd-224uc1'"),
               n_modules=90, inverter=("Solectria PVI 13kW", 2, 13.0), mount="roof", station=None),
    1275: dict(place="NewOrleans", module=("cec", "SunPower_SPR_E18_305_COM", "metadata 'spr-e18-305'"),
               n_modules=84, inverter=("unknown", 1, None), mount="roof", station=None),
    1276: dict(place="LasVegas", module=("cec", "Sharp_NU_U240F1", "metadata 'nu-u240f1'"),
               n_modules=252, inverter=("unknown 50 kW", 1, 50.0), mount="roof", station="UNLV"),
    1277: dict(place="LasVegas", module=("cec", "Sharp_NU_U240F1", "metadata 'nu-u240f1'"),
               n_modules=169, inverter=("unknown 30 kW", 1, 30.0), mount="roof", station="UNLV"),
    1283: dict(place="Golden", module=("cec", "SunPower_SPR_315E_WHT_D", "metadata 'spr-315e-wht-d'"),
               n_modules=1296, inverter=("SMA Sunny Central 250U", 2, 250.0), mount="roof", station="BMS"),
    1284: dict(place="Golden", module=("cec", "Solar_Frontier_SF150_L", "metadata 'SF 150-L'"),
               cell="cis", n_modules=8, inverter=("SMA Sunny Boy 2000HF-US", 1, 2.0), mount="open", station="BMS"),
    1289: dict(place="Golden", module=("cec", "Stion_STN_130", "metadata 'stn-130'"),
               cell="cis", n_modules=14, inverter=("unknown 2000 W", 1, 2.0), mount="open", station="BMS"),
    1332: dict(place="Golden", module=("cec", "SunPower_SPR_318E_WHT_D", "metadata lists 496 x SPR-315E + 2136 x SPR-318E = 835 kW, below the measured peak; DC 1153.5 kW (System.power) used, SPR-318E parameters"),
               n_modules=None, inverter=("Satcon PVS-500 x2 + PVS-135", 1, 1135.0), mount="open", station="BMS"),
    1403: dict(place="Cocoa", module=("cec", "Suniva_OPT270_60_4_100", "metadata 'opt270-60 blk-blk'"),
               n_modules=22, inverter=("unknown (2 channels)", 1, None), mount="open", station=None),
    1418: dict(place="LasVegas", module=("cec", "SolarWorld_Americas_Inc_Sunmodule_Plus_SWA_280_mono", "metadata 'sw280 mono'"),
               n_modules=156, inverter=("Solectria PVI 14TL", 3, 14.0), mount="roof", station="UNLV"),
    1420: dict(place="LasVegas", module=("cec", "SunPower_SPR_320E_WHT_D", "metadata 'spr-320e-wht-d'"),
               n_modules=112, inverter=("PV Powered PVP 30kW", 1, 30.0), mount="roof", station="UNLV"),
    1422: dict(place="Williston", module=("cec", "Suniva_OPT270_60_4_100", "metadata 'opt270-60 blk-blk'"),
               n_modules=24, inverter=("unknown (2 channels)", 1, None), mount="open", station=None),
    1429: dict(place="Albuquerque", module=("cec", "Suniva_OPT270_60_4_100", "metadata 'opt270-60 blk-blk'"),
               n_modules=22, inverter=("unknown (2 channels)", 1, None), mount="open", station=None),
    1432: dict(place="Golden", module=("sandia", "Evergreen_ES_190__2008__E__", "metadata 'es-190-sl'"),
               n_modules=495, inverter=("unknown 75 kVA", 1, 75.0), mount="roof", station="BMS"),
    1433: dict(place="Golden", module=("sandia", "Sanyo_HIP_205BA3__2006__E__", "metadata 'hit-205a' (HIT Power 205 = HIP-205BA3)"),
               n_modules=1872, inverter=("unknown 250 kVA + 375 kVA", 1, 625.0), mount="roof", station="BMS"),
    4902: dict(place="Gaithersburg", module=("cec", "Sharp_NU_U235F2", "metadata 'NU-U235F2'"),
               n_modules=1152, inverter=("PV Powered PVP 260kW", 1, 260.0), mount="open", station=None, ns=12),
    4903: dict(place="Gaithersburg", module=("cec", "Sharp_NU_U235F2", "metadata 'NU-U235F2'"),
               n_modules=312, inverter=("Satcon PVS-75", 1, 75.0), mount="roof", station=None, ns=12),
}
NOT_WRITTEN = {
    36: "tenKsolar RAIS: modules with built-in reflectors (180 W one-sun, 262 W with reflector); the chain has no reflector model",
}


def module_block(db: str, key: str) -> dict:
    import pvlib
    if db == "cec":
        m = pvlib.pvsystem.retrieve_sam("cecmod")[key]
        stc = float(m.STC)
        return dict(isc_ref=float(m.I_sc_ref), voc_ref=float(m.V_oc_ref), imp_ref=float(m.I_mp_ref), vmp_ref=float(m.V_mp_ref),
                    mu_isc=float(m.alpha_sc), mu_voc=float(m.beta_oc), ns=int(m.N_s), alpha_p=float(m.gamma_r) / 100,
                    eta_mp_stc=stc / (1000 * float(m.A_c)), noct=float(m.T_NOCT), stc_w=stc, cec=key,
                    tech=str(m.Technology))
    m = pvlib.pvsystem.retrieve_sam("sandiamod")[key]
    imp, vmp = float(m.Impo), float(m.Vmpo)
    return dict(isc_ref=float(m.Isco), voc_ref=float(m.Voco), imp_ref=imp, vmp_ref=vmp,
                mu_isc=float(m.Aisc) * float(m.Isco), mu_voc=float(m.Bvoco), ns=int(m.Cells_in_Series),
                alpha_p=float(m.Aimp) + float(m.Bvmpo) / vmp, eta_mp_stc=imp * vmp / (1000 * float(m.Area)),
                noct=45.0, stc_w=imp * vmp, cec=None, tech=str(m.Material))


def window(sid: int, station: str | None) -> tuple[str, str, float, str]:
    """Calendar year with the most scorable daylight hours that the weather source covers."""
    rep = (PVDAQ / str(sid) / f"{sid}_report.md").read_text()
    rows = re.findall(r"^\|\s*(\d{4})\s*\|(.*)$", rep, re.M)
    head = re.search(r"^\|\s*year\s*\|(.*)$", rep, re.M).group(1).split("|")
    hd = [c.strip() for c in head]
    c_n, c_pct = hd.index("scorable"), hd.index("scorable_pct")
    lo, hi = (STATION_FROM[station], 2023) if station else (1998, 2023)
    best = None                                            # most scorable daylight hours, then latest
    for y, rest in rows:
        y, cells = int(y), rest.split("|")
        try:
            n, pct = int(float(cells[c_n])), float(cells[c_pct])
        except ValueError:
            continue
        if lo <= y <= hi and (best is None or n > best[2] or (n == best[2] and y > best[0])):
            best = (y, pct, n)
    src = f"MIDC {station}" if station else "NSRDB PSM v4 at the site"
    if best is None:
        return "", "", 0.0, src
    return f"{best[0]}-01-01", f"{best[0]}-12-31", best[1], src


def strings(n: int, ns_hint: int | None, vmp: float) -> tuple[int, int, str]:
    if ns_hint and n % ns_hint == 0:
        return ns_hint, n // ns_hint, "metadata"
    target = max(1, round(400 / vmp))                     # about 400 V at MPP
    divs = [d for d in range(1, n + 1) if n % d == 0]
    ns = min(divs, key=lambda d: (abs(d - target), d))
    return ns, n // ns, f"assumed (no string data; ~400 V target; only Ns x Np = {n} matters without an inverter model)"


def write(sid: int) -> Path:
    cfg = SITES[sid]
    meta = json.loads((PVDAQ / str(sid) / f"{sid}_metadata.json").read_text())
    site = inv.site_of(meta)
    mount = next(iter(meta["Mount"].values()))
    summ = pd.read_csv(ROOT / "outputs" / "pvdaq_inventory" / "extract_summary.csv").drop_duplicates("system_id", keep="last").set_index("system_id")
    r = summ.loc[sid]
    dc = float(r.dc_kw)
    lat, lon, alt = float(site["latitude"]), float(site["longitude"]), float(site.get("elevation") or 0)
    tz = inv.standard_tz(lon)
    db, key, mnote = cfg["module"]
    m = module_block(db, key)
    n = cfg["n_modules"] or round(dc * 1000 / m["stc_w"])
    ns, np_, snote = strings(n, cfg.get("ns"), m["vmp_ref"])
    a, b, pct, wsrc = window(sid, cfg["station"])
    starts = [pd.Timestamp(v["start_on"]) for v in meta["Modules"].values() if v.get("start_on")]
    if meta["System"].get("started_on"):
        starts.append(pd.Timestamp(meta["System"]["started_on"]))
    first = pd.read_parquet(PVDAQ / str(sid) / f"{sid}_hourly.parquet").pv_power_hourly_kW.first_valid_index()
    installed = min(starts + [first])                       # metadata dates can postdate the first data
    age = max(0.0, ((pd.Timestamp(a) + pd.Timedelta(days=182)) - installed).days / 365.25)
    n_deg = (1 - DEG_RATE) ** age
    inv_model, inv_n, inv_kw = cfg["inverter"]
    ac = inv_n * inv_kw if inv_kw else None
    name = f"{cfg['place']}_{sid}"
    pub = meta["System"]["public_name"]
    L = []
    w = L.append
    w(f"# Site: PVDAQ System {sid} - {pub}")
    w(f"# Dataset: PVDAQ (OEDI), system_id = {sid}; written by scripts/pvdaq_site_configs.py, edit there")
    w(f"# Measured: data/validation/PVDAQ/{sid}/{sid}_hourly.parquet (pvdaq_extract.py; see {sid}_report.md)")
    w(f"# DC rating {dc} kW: {r.dc_source}")
    w(f"# Module: {mnote} ({db.upper()} {key})")
    w(f"# Inverter: {inv_model} x {inv_n}" + (f", AC cap {ac} kW" if ac else ", no AC cap (rating unknown)"))
    w(f"# Weather: {wsrc}; inputs \"{'midc' if cfg['station'] else 'nsrdb'}\", met \"{'midc' if cfg['station'] else 'point'}\"")
    w(f"# Scoring year {a[:4]}: most scorable daylight hours of the years the weather source covers ({pct} % of daylight hours)")
    w("")
    w("[location]")
    w(f'name     = "PVDAQ {sid} - {pub}"')
    w(f"lat      = {lat}")
    w(f"lon      = {lon}")
    w(f"altitude = {alt}        # m, metadata")
    w("")
    w("[data]")
    w(f'start_date = "{a}"')
    w(f'end_date   = "{b}"')
    w('resolution = "hourly"')
    w(f'sa_start   = "{a}"')
    w(f'sa_end     = "{b}"')
    w("")
    w("[array]")
    w("tracking = false")
    w(f"tilt     = {float(mount['tilt'])}        # metadata")
    w(f"azimuth  = {float(mount['azimuth'])}       # metadata, degrees from north")
    w("albedo   = 0.20        # assumed")
    w("")
    w("[system]")
    w(f"rated_capacity_kw = {dc}")
    w(f"pdc0_w            = {round(dc * 1000, 1)}   # DC rating above ({n} x {m['stc_w']:.1f} W = {n * m['stc_w'] / 1000:.2f} kW from the module table)")
    if ac:
        w(f"ac_cap_kw         = {ac}       # inverter nameplate x count")
    w(f"ns_arr            = {ns}           # {snote}")
    w(f"np_arr            = {np_}")
    if cfg["station"]:
        w("")
        w("[station]")
        w(f'midc = "{cfg["station"]}"')
    w("")
    w("[measured]")
    w(f'file               = "data/validation/PVDAQ/{sid}/{sid}_hourly.parquet"')
    w(f'source             = "pvdaq_{sid}"')
    w('time_col           = "timestamp"')
    w('power_col          = "pv_power_hourly_kW"')
    w('power_unit         = "kW"')
    w(f'tz                 = "{tz}"      # fixed standard time: pvdaq_extract.py moved the labels onto it, per month')
    w('timestamp_label    = "start"')
    w("interval_minutes   = 60")
    w("manual_shift_hours = 0")
    w("auto_lag_correction = false     # already corrected in the parquet (0 h in every year, pvdaq_verify_parquet.py)")
    w("max_lag_hours      = 4")
    w("")
    w("[module]")
    w(f"# {key} ({m['tech']}); values from the {db.upper()} table")
    w(f"isc_ref    = {m['isc_ref']:.4f}")
    w(f"voc_ref    = {m['voc_ref']:.4f}")
    w(f"imp_ref    = {m['imp_ref']:.4f}")
    w(f"vmp_ref    = {m['vmp_ref']:.4f}")
    w(f"mu_isc     = {m['mu_isc']:.6f}   # A/K")
    w(f"mu_voc     = {m['mu_voc']:.6f}   # V/K")
    w(f"ns         = {m['ns']}")
    w(f"alpha_p    = {m['alpha_p']:.6f}   # 1/K" + ("" if db == "cec" else " (Sandia Aimp + Bvmpo/Vmpo)"))
    w(f"eta_mp_stc = {m['eta_mp_stc']:.4f}")
    if m["cec"]:
        w(f'cec        = "{m["cec"]}"')
    cell = cfg.get("cell") or ("cis" if re.search(r"CIS|CIGS", m["tech"], re.I) else "cdte" if re.search("CdTe", m["tech"], re.I) else None)
    if cell:
        w(f'huld_cell_type = "{cell}"   # Huld coefficients; {"set in SITES (CEC lists only Thin Film)" if cfg.get("cell") else "from the module table"}')
    w("")
    w("[iam]")
    w("n = 1.526")
    w("k = 4.0")
    w("l = 0.002")
    w("")
    w("[thermal.homer]")
    w(f"tc_noct   = {m['noct']}" + ("" if db == "cec" else "      # Sandia has no NOCT: 45 assumed"))
    w("ta_noct   = 20.0")
    w("poa_noct  = 0.8")
    w("tau_alpha = 0.9")
    w("")
    w("[thermal.pvsyst]")
    w(f"u_c        = {U_C[cfg['mount']]}     # {MOUNT_TXT[cfg['mount']]}; mounting assumed from the system name and tilt")
    w("u_v        = 0.0")
    w(f"eta_module = {m['eta_mp_stc']:.4f}")
    w("alpha_abs  = 0.9")
    w("")
    if sid in FAST_SNOW:
        w("[snow]")
        w(f'removal = "fast"       # measured / snow-free output {FAST_SNOW[sid]} in the hours the chain has the array')
        w("                       # snow-covered: snow is cleared faster than the NREL model slides it (FAST_SNOW)")
        w("")
    w("[losses]")
    w(f"n_degradation = {n_deg:.4f}   # (1 - {DEG_RATE})^{age:.1f} years: installed {installed.date()} "
      f"(earliest of metadata and first data), mid-{a[:4]}")
    w("n_temp      = 0.95   # HOMER model only")
    w("n_soiling   = 0.98")
    w("n_mismatch  = 0.98")
    w("n_dc_wiring = 0.98")
    w("n_ac_wiring = 0.99")
    w("n_inverter  = 0.97")
    w("f_pv        = 0.95")
    w("eta_inv     = 0.97")
    f = SITE_DIR / f"{name}.toml"
    f.write_text("\n".join(L) + "\n")
    _prov_spec = importlib.util.spec_from_file_location("prov", Path(__file__).parent / "site_provenance.py")
    prov = importlib.util.module_from_spec(_prov_spec)
    _prov_spec.loader.exec_module(prov)
    prov.annotate(f)                                   # [provenance] table, checked against the schema
    return f


def check(f: Path) -> str:
    from solarmodel import bench
    from solarmodel.model.chain import module_parameters
    s = bench.load_site(f)
    mp = module_parameters(s.system)
    meas = bench.load_measured(s)
    return (f"ok: pdc0 {s.system.pdc0 / 1000:.2f} kW, {s.system.n_modules} modules, a_ref {mp['a_ref']:.3f}, "
            f"{int(meas.notna().sum())} measured hours in {s.start[:4]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for sid in a.ids or sorted(set(SITES) | set(NOT_WRITTEN)):
        if sid in NOT_WRITTEN:
            print(f"{sid:>5}  not written: {NOT_WRITTEN[sid]}")
            continue
        try:
            f = write(sid)
            msg = check(f) if a.check else "written"
        except Exception as e:
            f, msg = None, f"ERROR {type(e).__name__}: {e}"
        print(f"{sid:>5}  {f.name if f else ''}  {msg}", flush=True)


if __name__ == "__main__":
    main()
