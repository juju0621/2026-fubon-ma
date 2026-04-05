#!/bin/bash

FILEPATH="data.csv"

OUTPUT_DIR="output/rollover_analysis"

python rollover_analysis.py "$FILEPATH" \
    --output-dir "$OUTPUT_DIR" \