"""raw CSV -> curated Parquet, one symbol x month per run.

spark-submit spark/curation.py --raw s3://BUCKET/raw --out s3://BUCKET/curated --symbol LINKUSDT --month 2024-01

Output
  {out}/bookTicker/symbol=S/month=M/date=D/  ts_ms, ts, update_id, bid_px, bid_qty, ask_px, ask_qty
  {out}/aggTrades/symbol=S/month=M/date=D/   ts_ms, ts, agg_trade_id, price, qty, is_buyer_maker
  {out}/gaps/symbol=S/month=M/               gap_start, gap_end, seconds   (bookTicker silent > 60 s)
  {out}/counts/symbol=S/month=M/             feed, rows_in, rows_out
Each output gets _SUCCESS; a rerun skips any part that already has it.
"""
import argparse

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (BooleanType, DoubleType, LongType, StructField,
                               StructType)

BOOK = StructType([
    StructField("update_id", LongType()), StructField("best_bid_price", DoubleType()),
    StructField("best_bid_qty", DoubleType()), StructField("best_ask_price", DoubleType()),
    StructField("best_ask_qty", DoubleType()), StructField("transaction_time", LongType()),
    StructField("event_time", LongType())])
AGG = StructType([
    StructField("agg_trade_id", LongType()), StructField("price", DoubleType()),
    StructField("quantity", DoubleType()), StructField("first_trade_id", LongType()),
    StructField("last_trade_id", LongType()), StructField("transact_time", LongType()),
    StructField("is_buyer_maker", BooleanType())])
GAP_S = 60


def done(spark, path):
    jvm = spark._jvm
    p = jvm.org.apache.hadoop.fs.Path(path + "/_SUCCESS")
    return p.getFileSystem(spark._jsc.hadoopConfiguration()).exists(p)


def with_time(df, col):
    return (df.withColumnRenamed(col, "ts_ms")
              .withColumn("ts", F.timestamp_millis("ts_ms"))
              .withColumn("date", F.date_format(F.col("ts"), "yyyy-MM-dd")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--month", required=True)
    a = ap.parse_args()
    S, M = a.symbol, a.month
    spark = (SparkSession.builder.appName(f"curation-{S}-{M}")
             .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    part = f"symbol={S}/month={M}"
    counts = []

    # bookTicker: files are not in time order, and update_id can repeat
    out_b = f"{a.out}/bookTicker/{part}"
    if done(spark, out_b):
        print(f"SKIP {out_b}")
    else:
        raw = spark.read.csv(f"{a.raw}/bookTicker/{part}/", header=True, schema=BOOK)
        b = (with_time(raw, "transaction_time")
             .dropna(subset=["ts_ms", "best_bid_price", "best_ask_price"])
             .dropDuplicates(["update_id"])
             .select("ts_ms", "ts", "update_id",
                     F.col("best_bid_price").alias("bid_px"), F.col("best_bid_qty").alias("bid_qty"),
                     F.col("best_ask_price").alias("ask_px"), F.col("best_ask_qty").alias("ask_qty"),
                     "date"))
        (b.repartition("date").sortWithinPartitions("ts_ms", "update_id")
          .write.mode("overwrite").partitionBy("date").parquet(out_b))
        b2 = spark.read.parquet(out_b)
        counts.append(("bookTicker", raw.count(), b2.count()))
        w = Window.orderBy("ts_ms")
        gaps = (b2.select("ts_ms").distinct()
                .withColumn("prev", F.lag("ts_ms").over(w))
                .where(F.col("ts_ms") - F.col("prev") > GAP_S * 1000)
                .select(F.timestamp_millis("prev").alias("gap_start"),
                        F.timestamp_millis("ts_ms").alias("gap_end"),
                        ((F.col("ts_ms") - F.col("prev")) / 1000).alias("seconds")))
        gaps.coalesce(1).write.mode("overwrite").parquet(f"{a.out}/gaps/{part}")
        print(f"GAPS {S} {M}: {gaps.count()}")

    out_a = f"{a.out}/aggTrades/{part}"
    if done(spark, out_a):
        print(f"SKIP {out_a}")
    else:
        raw = spark.read.csv(f"{a.raw}/aggTrades/{part}/", header=True, schema=AGG)
        t = (with_time(raw, "transact_time")
             .dropna(subset=["ts_ms", "price", "quantity"])
             .dropDuplicates(["agg_trade_id"])
             .select("ts_ms", "ts", "agg_trade_id", "price",
                     F.col("quantity").alias("qty"), "is_buyer_maker", "date"))
        (t.repartition("date").sortWithinPartitions("ts_ms", "agg_trade_id")
          .write.mode("overwrite").partitionBy("date").parquet(out_a))
        counts.append(("aggTrades", raw.count(), spark.read.parquet(out_a).count()))

    if counts:
        c = spark.createDataFrame(counts, "feed string, rows_in long, rows_out long")
        c.coalesce(1).write.mode("append").parquet(f"{a.out}/counts/{part}")
        for r in counts:
            print(f"COUNT {S} {M} {r[0]} in={r[1]} out={r[2]}")
    spark.stop()


if __name__ == "__main__":
    main()
