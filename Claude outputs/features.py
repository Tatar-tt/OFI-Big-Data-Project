"""curated Parquet -> features, 1 row per symbol x second. One symbol per run, all months.

spark-submit spark/features.py --curated s3://BUCKET/curated --out s3://BUCKET/features --symbol LINKUSDT \
    --train-month 2024-01 --test-month 2024-02

OFI per bookTicker update (Cont, Kukanov, Stoikov 2014):
  e_n = 1[Pb_n >= Pb_n-1] Qb_n - 1[Pb_n <= Pb_n-1] Qb_n-1 - 1[Pa_n <= Pa_n-1] Qa_n + 1[Pa_n >= Pa_n-1] Qa_n-1
Summed per second, then over trailing windows (ts - w, ts].
Output columns follow the board's schema; written to {out}/symbol=S/month=M/.
"""
import argparse

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

WINDOWS = [1, 5, 10, 30]
VOL_S = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--train-month", default="2024-01")
    ap.add_argument("--test-month", default="2024-02")
    a = ap.parse_args()
    S = a.symbol
    spark = (SparkSession.builder.appName(f"features-{S}")
             .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    months = [a.train_month, a.test_month]

    book = (spark.read.parquet(f"{a.curated}/bookTicker/symbol={S}")
            .where(F.col("month").isin(months)))
    trades = (spark.read.parquet(f"{a.curated}/aggTrades/symbol={S}")
              .where(F.col("month").isin(months)))

    # 1. OFI per update. Partition by date so no single executor holds a whole month.
    wu = Window.partitionBy("date").orderBy("ts_ms", "update_id")
    pb, qb, pa, qa = (F.lag(c).over(wu) for c in ("bid_px", "bid_qty", "ask_px", "ask_qty"))
    e = (F.when(F.col("bid_px") >= pb, F.col("bid_qty")).otherwise(0)
         - F.when(F.col("bid_px") <= pb, qb).otherwise(0)
         - F.when(F.col("ask_px") <= pa, F.col("ask_qty")).otherwise(0)
         + F.when(F.col("ask_px") >= pa, qa).otherwise(0))
    upd = book.withColumn("e", e).withColumn("sec", F.floor(F.col("ts_ms") / 1000).cast("long"))

    # 2. Per second: OFI sum, last quote of the second
    last = F.struct("ts_ms", "update_id")
    bsec = upd.groupBy("sec").agg(
        F.sum(F.coalesce("e", F.lit(0.0))).alias("ofi"),
        F.max_by("bid_px", last).alias("bid"),
        F.max_by("ask_px", last).alias("ask"))
    tsec = trades.withColumn("sec", F.floor(F.col("ts_ms") / 1000).cast("long")).groupBy("sec").agg(
        F.sum(F.when(~F.col("is_buyer_maker"), F.col("qty")).otherwise(0)).alias("taker_buy_vol"),
        F.sum(F.when(F.col("is_buyer_maker"), F.col("qty")).otherwise(0)).alias("taker_sell_vol"),
        F.count("*").alias("n_trades"))

    # 3. Dense 1 s grid over the data span, so windows count seconds, not updates
    lo, hi = bsec.agg(F.min("sec"), F.max("sec")).first()
    grid = spark.range(lo, hi + 1).withColumnRenamed("id", "sec")
    g = (grid.join(bsec, "sec", "left").join(tsec, "sec", "left")
         .withColumn("ts", F.timestamp_seconds(F.col("sec") + 1))   # bar close
         .withColumn("date", F.date_format(F.timestamp_seconds("sec"), "yyyy-MM-dd"))
         .withColumn("month", F.date_format(F.timestamp_seconds("sec"), "yyyy-MM"))
         .fillna(0, ["ofi", "taker_buy_vol", "taker_sell_vol", "n_trades"]))

    wd = Window.partitionBy("date").orderBy("sec")
    ff = wd.rowsBetween(Window.unboundedPreceding, 0)
    g = (g.withColumn("bid", F.last("bid", ignorenulls=True).over(ff))
          .withColumn("ask", F.last("ask", ignorenulls=True).over(ff))
          .withColumn("mid", (F.col("bid") + F.col("ask")) / 2)
          .withColumn("spread_bps", (F.col("ask") - F.col("bid")) / F.col("mid") * 1e4)
          .withColumn("ret_1s", (F.col("mid") / F.lag("mid").over(wd) - 1) * 1e4))
    for w in WINDOWS:
        g = (g.withColumn(f"ofi_{w}s", F.sum("ofi").over(wd.rowsBetween(-(w - 1), 0)))
              .withColumn(f"fwd_ret_{w}s", (F.lead("mid", w).over(wd) / F.col("mid") - 1) * 1e4))
    g = g.withColumn("rv", F.stddev("ret_1s").over(wd.rowsBetween(-(VOL_S - 1), 0)))

    # 4. Volatility regime: thresholds from the train month only
    q1, q2 = g.where(F.col("month") == a.train_month).approxQuantile("rv", [1 / 3, 2 / 3], 0.001)
    print(f"VOL {S} thresholds q33={q1:.4f} q67={q2:.4f}")
    g = (g.withColumn("vol_regime", F.when(F.col("rv").isNull(), None)
                      .when(F.col("rv") <= q1, "low").when(F.col("rv") <= q2, "mid").otherwise("high"))
          .withColumn("split", F.when(F.col("month") == a.train_month, "train").otherwise("test"))
          .withColumn("hour", F.hour(F.timestamp_seconds("sec")))
          .withColumn("symbol", F.lit(S))
          .where(F.col("mid").isNotNull()))

    cols = (["symbol", "ts", "date", "split", "mid", "ret_1s", "spread_bps"]
            + [c for w in WINDOWS for c in (f"ofi_{w}s", f"fwd_ret_{w}s")]
            + ["taker_buy_vol", "taker_sell_vol", "n_trades", "hour", "vol_regime", "month"])
    out = g.select(*cols).withColumn("n_trades", F.col("n_trades").cast("int"))
    (out.repartition("month").sortWithinPartitions("ts")
        .write.mode("overwrite").partitionBy("month").parquet(f"{a.out}/symbol={S}"))
    print(f"FEATURES {S} rows={spark.read.parquet(f'{a.out}/symbol={S}').count()}")
    spark.stop()


if __name__ == "__main__":
    main()
