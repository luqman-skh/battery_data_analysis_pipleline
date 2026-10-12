#!/bin/sh

# Run the pytest command
python -m pytest test_pipeline.py -v

# Run the metaflow pipeline
python battery_flow.py run

# Run the mlflow UI
mlflow ui --host 0.0.0.0
