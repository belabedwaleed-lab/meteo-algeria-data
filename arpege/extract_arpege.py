#!/usr/bin/env python3
"""
استخراج بيانات ARPEGE 0.1° (Météo-France) لمدن الجزائر وكتابتها في arpege.json
بنفس شكل أيام/ساعات الـ Worker (days / hours) ليسهل الدمج مع MET Norway.

المصدر: ملفات GRIB2 المفتوحة (Licence Ouverte 2.0) عبر مكتبة meteofetch، بدون مفتاح.
الإسناد المطلوب: "Source : Météo-France".

الاستعمال:
    python extract_arpege.py --discover      # يطبع أسماء الحقول المتاحة فعلاً (شغّله مرة أولى)
    python extract_arpege.py --out arpege.json

ملاحظات:
  - شبكة EURAT01 تغطي 20N–72N و32W–42E. مدينة "عين قزام" (19.57N) خارجها فتُترك لـ MET وحده.
  - أسماء الحقول أدناه مرشّحة (CANDIDATES) وتُختار أول ما يوجد؛ إن اختلفت عندك
    فالأمر --discover سيريك الأسماء الحقيقية لتعديل القاموس.
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

TZ_HOURS = 1  # الجزائر UTC+1
# عتبات إشارة "احتمال رعد": تقدير قابل للضبط (لم تُعايَر بقياسات)
CAPE_MIN = 500.0   # طاقة الحمل (جول/كغ) على الأقل
CP_MIN = 0.3       # معدل الهطول الحملي (ملم/س) على الأقل
DOMAIN = {"lat_min": 20.0, "lat_max": 72.0, "lon_min": -32.0, "lon_max": 42.0}

CITIES = [
    ("الجزائر", 36.7538, 3.0588), ("تنس", 36.5104, 1.3083), ("وهران", 35.6971, -0.6308),
    ("قسنطينة", 36.365, 6.6147), ("تيارت", 35.3711, 1.317), ("جيجل", 36.8206, 5.7667),
    ("عنابة", 36.9, 7.7667), ("تلمسان", 34.8783, -1.315), ("بشار", 31.6167, -2.2167),
    ("تندوف", 27.6711, -8.1474), ("أدرار", 27.8742, -0.2939), ("تمنراست", 22.785, 5.5228),
    ("جانت", 24.5533, 9.4858), ("إليزي", 26.4833, 8.4667), ("ورقلة", 31.9493, 5.325),
    ("المنيعة", 30.5811, 2.8836), ("الوادي", 33.3683, 6.8674), ("سطيف", 36.1911, 5.4137),
    ("الجلفة", 34.6667, 3.25), ("عين صالح", 27.195, 2.4833), ("عين قزام", 19.5686, 5.7722),
    ("برج باجي مختار", 21.3283, 0.9432),
]

# paquet -> {مفهوم: [أسماء مرشحة]}
CANDIDATES = {
    "SP1": {
        "t2m": ["t2m", "2t"], "u10": ["u10", "10u"], "v10": ["v10", "10v"],
        "gust": ["fg10", "10fg", "i10fg"], "precip": ["tp", "tirf", "tprate"],
        "cloud": ["tcc"], "rh": ["r2", "2r"], "pres": ["msl", "prmsl"],
        "cp": ["cp"], "snow": ["tsnowp"],
    },
    "SP2": {"d2m": ["d2m", "2d"], "sp": ["sp"], "lcc": ["lcc"], "mcc": ["mcc"], "hcc": ["hcc"], "cape": ["CAPE_INS"], "ptype": ["ptype"]},
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- فيزياء
def apparent_temp(t, rh, ws):
    e = (rh / 100.0) * 6.105 * np.exp((17.27 * t) / (237.7 + t))
    return t + 0.33 * e - 0.7 * ws - 4.0


def rh_from_dewpoint(t, td):
    a, b = 17.625, 243.04
    return 100.0 * np.exp((a * td) / (b + td) - (a * t) / (b + t))


def normalize(concept, s):
    """توحيد الوحدات: حرارة °C، ضغط hPa، غيوم %، رطوبة %."""
    s = s.astype(float)
    if concept in ("t2m", "d2m") and np.nanmean(s) > 150:
        s = s - 273.15
    if concept in ("pres", "sp") and np.nanmean(s) > 2000:
        s = s / 100.0
    return s


# ---------------------------------------------------------------- القراءة
def pick(dsdict, names):
    for n in names:
        if n in dsdict:
            return n, dsdict[n]
    return None, None


def fetch_fields(discover=False):
    """يعيد {مفهوم: DataArray(time, latitude, longitude)}"""
    from meteofetch import Arpege01

    out = {}
    for paquet, concepts in CANDIDATES.items():
        wanted = sorted({n for names in concepts.values() for n in names})
        log(f"تنزيل {paquet} ...")
        t0 = time.time()
        ds = Arpege01.get_latest_forecast(paquet=paquet, variables=None if discover else wanted)
        log(f"  تم في {time.time() - t0:.0f} ثانية، الحقول: {sorted(ds.keys())}")
        if discover:
            for k, v in ds.items():
                log(f"   - {k}: dims={v.dims} shape={v.shape} attrs={ {a: v.attrs.get(a) for a in ('long_name','units','GRIB_stepType')} }")
            continue
        for concept, names in concepts.items():
            if concept in out:
                continue  # الأولوية لـ SP1
            name, da = pick(ds, names)
            if da is not None:
                log(f"  {paquet}.{name} -> {concept}")
                out[concept] = da
    return out


# ---------------------------------------------------------------- البناء
def city_series(fields, lat, lon):
    ser = {}
    for concept, da in fields.items():
        pt = da.sel(latitude=lat, longitude=lon, method="nearest")
        s = pd.Series(np.asarray(pt.values, dtype=float), index=pd.to_datetime(pt["time"].values))
        ser[concept] = normalize(concept, s.sort_index())
    return ser


def precip_per_step(s):
    """تراكمي منذ بداية التشغيلة؟ نحوّله إلى كمية كل خطوة، وإلا نعتبره لكل خطوة."""
    d = s.diff()
    if (d.dropna() >= -0.05).all() and s.iloc[-1] > 0:
        log("  الهطول: تراكمي (يُشتق الفرق بين الخطوات)")
        d.iloc[0] = s.iloc[0]
        return d.clip(lower=0)
    return s.clip(lower=0)


def per_hour(s):
    """هطول تراكمي → معدل ملم/ساعة لكل خطوة."""
    d = precip_per_step(s)
    h = pd.Series(s.index, index=s.index).diff().dt.total_seconds().div(3600).bfill().clip(lower=1)
    return d / h


def ptype_codes(s, key):
    """القيم المختلفة لحقل نوع الهطول في يوم ما (للتعلّم من ترميزه)."""
    if s is None:
        return None
    loc = (s.index + pd.Timedelta(hours=TZ_HOURS)).strftime("%Y-%m-%d")
    vals = s[loc == key].dropna()
    return sorted({round(float(v), 1) for v in vals})[:8]
def build_days(ser, now_utc):
    keys = [(now_utc + pd.Timedelta(hours=TZ_HOURS + 24 * o)).strftime("%Y-%m-%d") for o in range(3)]
    t = ser["t2m"]
    idx = t.index
    df = pd.DataFrame({"t": t})
    for c in ("u10", "v10", "gust", "cloud", "rh", "pres", "d2m", "sp", "lcc", "mcc", "hcc", "cape"):
        if c in ser:
            df[c] = ser[c].reindex(idx, method="nearest", tolerance=pd.Timedelta(hours=3))
    if "cloud" not in df and {"lcc", "mcc", "hcc"} <= set(df.columns):
        # تقدير الغيوم الكلية من الطبقات: متوسط حدّي التداخل (أقصى تداخل / تداخل عشوائي)
        l, m, h = (df[c].clip(0, 100) / 100.0 for c in ("lcc", "mcc", "hcc"))
        rand = 1 - (1 - l) * (1 - m) * (1 - h)
        mx = np.maximum(np.maximum(l, m), h)
        df["cloud"] = 100.0 * (rand + mx) / 2.0
    if "rh" not in df and "d2m" in df:
        df["rh"] = rh_from_dewpoint(df["t"], df["d2m"])
    if "pres" not in df and "sp" in df:
        df["pres"] = df["sp"]
    if "u10" in df and "v10" in df:
        df["ws"] = np.hypot(df["u10"], df["v10"])
        df["dir"] = (np.degrees(np.arctan2(-df["u10"], -df["v10"])) % 360)
    prec = precip_per_step(ser["precip"]) if "precip" in ser else None
    cp_rate = per_hour(ser["cp"]) if "cp" in ser else None
    snow_rate = per_hour(ser["snow"]) if "snow" in ser else None
    step_h = pd.Series(idx, index=idx).diff().dt.total_seconds().div(3600).bfill().clip(lower=1)

    df["local"] = idx + pd.Timedelta(hours=TZ_HOURS)
    df["key"] = df["local"].dt.strftime("%Y-%m-%d")
    days = []
    r1 = lambda v: None if v is None or pd.isna(v) else round(float(v), 1)
    for k in keys:
        d = df[df["key"] == k]
        if d.empty:
            continue
        feels = None
        if {"rh", "ws"} <= set(d.columns):
            feels = apparent_temp(d["t"], d["rh"], d["ws"])
        pday = None
        if prec is not None:
            loc = (prec.index + pd.Timedelta(hours=TZ_HOURS)).strftime("%Y-%m-%d")
            pday = prec[loc == k].sum()
        w = d["ws"] * 3.6 if "ws" in d else None
        g = (d["gust"] * 3.6) if "gust" in d else w
        hours = []
        for ts, row in d.iterrows():
            pr = None
            if prec is not None and ts in prec.index:
                pr = prec.loc[ts] / float(step_h.loc[ts])
            cape = float(row["cape"]) if "cape" in d and not pd.isna(row["cape"]) else None
            cpr = cp_rate.get(ts) if cp_rate is not None else None
            snr = snow_rate.get(ts) if snow_rate is not None else None
            thunder = bool(cape is not None and cpr is not None and cape >= CAPE_MIN and cpr >= CP_MIN)
            hours.append({
                "datetime": f"{row['local'].hour:02d}:00:00",
                "temp": r1(row["t"]),
                "precip": r1(pr) if pr is not None else None,
                "cloudcover": r1(row["cloud"]) if "cloud" in d else None,
                "windspeed": r1(row["ws"] * 3.6) if "ws" in d else None,
                "windgust": r1(row["gust"] * 3.6) if "gust" in d else None,
                "snow": r1(snr),
                "thunder": thunder,
                "cape": int(round(cape)) if cape is not None else None,
            })
        days.append({
            "datetime": k,
            "tempmax": r1(d["t"].max()), "tempmin": r1(d["t"].min()), "temp": r1(d["t"].mean()),
            "feelslikemax": r1(feels.max()) if feels is not None else None,
            "feelslikemin": r1(feels.min()) if feels is not None else None,
            "humidity": r1(d["rh"].mean()) if "rh" in d else None,
            "precip": r1(pday),
            "windspeed": r1(w.mean()) if w is not None else None,
            "windgust": r1(g.max()) if g is not None else None,
            "winddir": int(round(d["dir"].mean())) if "dir" in d else None,
            "pressure": r1(d["pres"].mean()) if "pres" in d else None,
            "cloudcover": r1(d["cloud"].mean()) if "cloud" in d else None,
            "cape_max": r1(d["cape"].max()) if "cape" in d else None,
            "thunder_hours": sum(1 for h in hours if h["thunder"]),
            "ptype_codes": ptype_codes(ser.get("ptype"), k),
            "hours": hours,
        })
    return days
                

def run(fields, now_utc=None):
    now_utc = now_utc or pd.Timestamp.utcnow().tz_localize(None)
    cities, skipped = {}, []
    for name, lat, lon in CITIES:
        if not (DOMAIN["lat_min"] <= lat <= DOMAIN["lat_max"] and DOMAIN["lon_min"] <= lon <= DOMAIN["lon_max"]):
            skipped.append(name)
            continue
        ser = city_series(fields, lat, lon)
        if name == "قسنطينة":
            p = ser.get("precip")
            log("DEBUG tp", len(p), [f"{t:%d %H}:{v:.1f}" for t, v in p.items()])
        if "t2m" not in ser:
            raise RuntimeError("حقل الحرارة غير موجود — شغّل --discover وعدّل CANDIDATES")
        days = build_days(ser, now_utc)
        if days:
            cities[name] = days
    return {
        "generated": int(time.time() * 1000),
        "model": "ARPEGE 0.1",
        "attribution": "Source : Météo-France (Licence Ouverte 2.0)",
        "skipped_outside_domain": skipped,
        "concepts": sorted(fields.keys()),
        "cities": cities,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--out", default="arpege.json")
    a = ap.parse_args()
    if a.discover:
        fetch_fields(discover=True)
        sys.exit(0)
    result = run(fetch_fields())
    if not result["cities"]:
        sys.exit("لا بيانات")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    log(f"كُتب {a.out}: {len(result['cities'])} مدينة، حقول: {result['concepts']}")
