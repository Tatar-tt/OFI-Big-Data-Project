# Order Flow Imbalance at Scale (DES431, Group 11)

Does order flow imbalance (OFI) predict short-horizon returns in crypto perpetual futures after fees and spread, and how much of any result is selection bias?

- Data: Binance USDⓈ-M `bookTicker` + `aggTrades`, LINKUSDT and DOGEUSDT, 2024-01 (train) and 2024-02 (test)
- Pipeline: EC2 → S3 raw → EMR Spark (`spark/curation.py`, `spark/features.py`) → S3 Parquet → Glue / Athena → dashboard
- Design and the 16 pre-registered specifications: [`prereg.md`](prereg.md)

## Start without real data
```bash
pip install pandas pyarrow
python tools/make_fake.py        # -> data/features_fake.parquet, same schema as the real features
```

Kick-off documents are in `docs/`.
