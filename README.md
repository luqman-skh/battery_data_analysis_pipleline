
# Battery Life Prediction Pipeline

This project implements a data preprocessing and machine learning pipeline for battery life prediction using XGBoost. The workflow is orchestrated with Metaflow and includes MLflow tracking.

## Requirements
- Python 3.8+
- [requirements.txt](./requirements.txt)

## Installation
```bash
pip install -r requirements.txt
```

## Usage

1. **Start MLflow Tracking Server**

   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   Keep this running in a separate terminal.

2. **Run the Pipeline**

   ```bash
   python battery_flow.py run \
     --filepath=/path/to/your_data.parquet \
     --mlflow_uri=http://localhost:5000
   ```
   Replace `/path/to/your_data.parquet` with your dataset path.

   Default dataset name: `case_study_sample_dataset` (ensure the file exists).

3. **Check Results**

   Model artifacts and metrics will be logged to MLflow at [http://localhost:5000](http://localhost:5000).

   Output plots (`histograms.png`, `final_scatter.png`) will be saved locally.

4. **Testing**

   Run unit and integration tests:

   ```bash
   python -m pytest test_pipeline.py -v
   ```

5. **Sphinx Documentation**

   ```bash
   sphinx-quickstart
   sphinx-apidoc -o docs your_script_directory
   cd docs
   make html
   ```
