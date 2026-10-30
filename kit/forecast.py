"""Build a wind-adjusted tide forecast for Charleston from NOAA predictions, recent observations and the NWS wind forecast."""
import json, re, urllib.request, datetime as dt, numpy as np, pandas as pd

UA = {"User-Agent": "charleston-fishing-planner", "Accept": "application/geo+json"}
CO = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
M = json.load(open("data/model.json"))
coef, W, tau = np.array(M["coef"]), M["windows"], M["tau"]

def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r: return json.load(r)
def co(**p):
    p = {"station": "8665530", "units": "english", "time_zone": "gmt", "format": "json", "application": "fishing_planner", **p}
    return get(CO + "?" + "&".join(f"{k}={str(v).replace(' ', '%20')}" for k, v in p.items()))

now = pd.Timestamp.now("UTC").tz_localize(None).floor("h")
b30 = (now - pd.Timedelta(days=31)).strftime("%Y%m%d %H:%M"); e0 = now.strftime("%Y%m%d %H:%M")

# --- recent observations (6-min preliminary -> hourly) ---
wl = pd.DataFrame(co(product="water_level", datum="MLLW", begin_date=b30, end_date=e0)["data"])
wl = pd.Series(pd.to_numeric(wl.v, errors="coerce").values, pd.to_datetime(wl.t)).resample("h").mean()
pr = pd.DataFrame(co(product="predictions", datum="MLLW", interval="h", begin_date=b30, end_date=e0)["predictions"])
pr = pd.Series(pr.v.astype(float).values, pd.to_datetime(pr.t))
resid = (wl - pr).dropna()
wn = pd.DataFrame(co(product="wind", interval="h", begin_date=(now - pd.Timedelta(days=3)).strftime("%Y%m%d %H:%M"), end_date=e0)["data"])
wn = pd.DataFrame({"spd": pd.to_numeric(wn.s, errors="coerce").values, "dir": pd.to_numeric(wn.d, errors="coerce").values}, index=pd.to_datetime(wn.t))
mb = pd.DataFrame(co(product="air_pressure", interval="h", begin_date=(now - pd.Timedelta(days=10)).strftime("%Y%m%d %H:%M"), end_date=e0)["data"])
mb = pd.Series(pd.to_numeric(mb.v, errors="coerce").values, pd.to_datetime(mb.t))

# --- NWS hourly wind forecast ---
pt = get("https://api.weather.gov/points/32.781,-79.925")
fh = get(pt["properties"]["forecastHourly"])["properties"]["periods"]
CMP = {k: i * 22.5 for i, k in enumerate("N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split())}
fw = []
for p in fh:
    mph = [float(x) for x in re.findall(r"\d+", p["windSpeed"])]
    t = pd.Timestamp(p["startTime"]).tz_convert("UTC").tz_localize(None)
    fw.append((t, np.mean(mph) * 0.869, CMP.get(p["windDirection"], np.nan)))
fw = pd.DataFrame(fw, columns=["t", "spd", "dir"]).set_index("t")
fcast_end = fw.index.max()

wind = pd.concat([wn[wn.index < fw.index.min()], fw]).sort_index()
wind = wind[~wind.index.duplicated()].asfreq("h").interpolate(limit=6)
th = np.deg2rad(wind.dir); u, v = -wind.spd * np.sin(th), -wind.spd * np.cos(th); m = np.hypot(u, v)
tu, tv = u * m / 10, v * m / 10

def wind_effect(t):
    x = [1.0]
    for a, b in W:
        sl = slice(t - pd.Timedelta(hours=b - 1), t - pd.Timedelta(hours=a))
        x += [tu[sl].mean(), tv[sl].mean()]
    return float(np.dot(coef[:-1], x))

# pressure anomaly now, decayed over 2 days going forward (no pressure in NWS hourly)
dp_now = float(mb.rolling(240, min_periods=48).mean().iloc[-1] - mb.iloc[-1]) if len(mb) > 48 else 0.0
base = float(resid.tail(24 * 30).mean())
t_last = resid.index.max()
w_last = wind_effect(t_last)
anom0 = float(resid.tail(3).mean()) - base - w_last - coef[-1] * dp_now

# --- future predictions (hourly, 30 days) ---
hor_end = now + pd.Timedelta(days=30)
pf = pd.DataFrame(co(product="predictions", datum="MLLW", interval="h", begin_date=now.strftime("%Y%m%d %H:%M"), end_date=hor_end.strftime("%Y%m%d %H:%M"))["predictions"])
pf = pd.Series(pf.v.astype(float).values, pd.to_datetime(pf.t))
hilo = co(product="predictions", datum="MLLW", interval="hilo", begin_date=now.strftime("%Y%m%d"), end_date=hor_end.strftime("%Y%m%d"))["predictions"]

RMSE = M["metrics"]
out = []
for t, p in pf.items():
    h = (t - t_last) / pd.Timedelta(hours=1)
    in_fc = t <= fcast_end
    wsurge = wind_effect(t) if in_fc else 0.0
    surge = base + wsurge + anom0 * np.exp(-h / tau) + coef[-1] * dp_now * np.exp(-h / 48)
    # error band: validated model error + growth for wind-forecast error; wider past the wind forecast
    band = (0.28 + 0.035 * h / 24) if in_fc else 0.45
    out.append({"t": t.strftime("%Y-%m-%dT%H:%MZ"), "pred": round(p, 2), "surge": round(surge, 2),
                "wind_part": round(wsurge, 2), "band": round(band, 2), "wind_fc": bool(in_fc)})

fwj = [{"t": t.strftime("%Y-%m-%dT%H:%MZ"), "kt": round(r.spd, 0), "dir": r.dir} for t, r in fw.iterrows()]
recent = [{"t": t.strftime("%Y-%m-%dT%H:%MZ"), "obs": round(float(wl[t]), 2), "pred": round(float(pr[t]), 2)}
          for t in resid.index[-72:]]
res = {"generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%MZ"), "base_offset": round(base, 2),
       "anom_now": round(anom0, 2), "resid_now": round(float(resid.tail(3).mean()), 2),
       "wind_fc_end": fcast_end.strftime("%Y-%m-%dT%H:%MZ"), "hourly": out, "wind": fwj, "recent": recent,
       "hilo": [{"t": x["t"], "v": float(x["v"]), "type": x["type"]} for x in hilo],
       "model": {k: M[k] for k in ("metrics", "steady", "resid_mean_2yr", "tau", "train_end")},
       "wind_obs_now": {"kt": float(wn.spd.dropna().iloc[-1]), "dir": float(wn.dir.dropna().iloc[-1])}}
import os
if os.path.exists("data/event.json"): res["event"] = json.load(open("data/event.json"))
json.dump(res, open("data/forecast.json", "w"), default=float)
print({k: res[k] for k in ("generated", "base_offset", "anom_now", "resid_now", "wind_fc_end", "wind_obs_now")})
print("hours", len(out), "hilo", len(hilo))
for r in out[:120:12]: print(r)
