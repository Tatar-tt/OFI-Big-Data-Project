# tools/make_fake.py : features ปลอมตาม schema จริง ใช้เขียน code ก่อนข้อมูลจริงมา
# รัน: python tools/make_fake.py  ->  data/features_fake.parquet  (ต้องมี pandas, pyarrow)
import os
import numpy as np, pandas as pd

rng = np.random.default_rng(431)
W = [1, 5, 10, 30]
DAYS = ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]   # 2 วัน train, 2 วัน test
out = []
for sym, px, tick in [("LINKUSDT", 15.0, 0.001), ("DOGEUSDT", 0.08, 0.00001)]:
    ts = pd.date_range(DAYS[0], periods=86400 * len(DAYS), freq="s", tz="UTC")
    n = len(ts)
    e = rng.standard_normal(n)                 # order flow ต่อวินาที
    r = 0.3 * e + rng.standard_normal(n)       # ret 1 s (bps): ขยับตาม OFI ในวินาทีเดียวกัน
    r[1:] += 0.05 * e[:-1]                     # สัญญาณทำนายล่วงหน้า (เล็กมาก)
    mid = pd.Series(px * np.exp(np.cumsum(r) / 1e4))
    df = pd.DataFrame({"symbol": sym, "ts": ts, "date": ts.strftime("%Y-%m-%d"),
                       "split": np.where(ts.month == 1, "train", "test"),
                       "mid": mid.values, "ret_1s": r})
    df["spread_bps"] = tick / df["mid"] * 1e4 * rng.choice([1, 1, 1, 2], n)
    for w in W:
        df[f"ofi_{w}s"] = pd.Series(e).rolling(w, min_periods=1).sum().values
        df[f"fwd_ret_{w}s"] = ((mid.shift(-w) / mid - 1) * 1e4).values
    df["taker_buy_vol"] = np.abs(e) * (e > 0) * 100 + rng.exponential(20, n)
    df["taker_sell_vol"] = np.abs(e) * (e < 0) * 100 + rng.exponential(20, n)
    df["n_trades"] = rng.poisson(3, n)
    df["hour"] = ts.hour
    vol = pd.Series(r).rolling(300, min_periods=30).std()
    lo, hi = vol[(df["split"] == "train").values].quantile([1 / 3, 2 / 3])   # threshold จาก train เท่านั้น
    df["vol_regime"] = np.select([vol <= lo, vol <= hi], ["low", "mid"], "high")
    out.append(df)

os.makedirs("data", exist_ok=True)
pd.concat(out, ignore_index=True).to_parquet("data/features_fake.parquet", index=False)
print("wrote data/features_fake.parquet")
