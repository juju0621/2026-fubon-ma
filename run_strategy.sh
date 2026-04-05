#!/bin/bash
CSV_PATH="data.csv"
OUT_DIR="output/hmm_regime"

python hmm_vol_regime_strategy.py \
    --csv "$CSV_PATH" \
    --outdir "$OUT_DIR" \