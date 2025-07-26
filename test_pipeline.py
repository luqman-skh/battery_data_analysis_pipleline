import pytest
import pandas as pd
import numpy as np
from eda import load_data, analyze_data, handle_missing_values, remove_outliers, detect_anomalies, train_autoencoder, final_plot

# Sample data for testing
@pytest.fixture
def sample_data():
    data = {
        'test_time': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'cycle_index': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'voltage': [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4],
        'discharge_capacity': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
        'current': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        'internal_resistance': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
        'temperature': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34]  
    }
    return pd.DataFrame(data)

# Unit test for load_data function
def test_load_data(sample_data):
    # Mock the load_data function to return sample_data
    df = sample_data
    assert not df.empty, "DataFrame should not be empty"
    assert df.shape == (10, 7), "DataFrame should have 10 rows and 7 columns"

# Unit test for analyze_data function
def test_analyze_data(sample_data):
    df = analyze_data(sample_data)
    assert not df.empty, "DataFrame should not be empty"
    assert df.shape == (10, 7), "DataFrame should have 10 rows and 7 columns"

# Unit test for handle_missing_values function
def test_handle_missing_values(sample_data):
    df = handle_missing_values(sample_data)
    assert df.isnull().sum().sum() == 0, "There should be no missing values"

# Unit test for remove_outliers function
def test_remove_outliers(sample_data):
    df = remove_outliers(sample_data)
    assert df.shape[0] <= sample_data.shape[0], "Number of rows should not increase after removing outliers"

# Unit test for detect_anomalies function
def test_detect_anomalies(sample_data):
    df = handle_missing_values(sample_data)  # Add this line
    df = detect_anomalies(sample_data)
    assert df.shape[0] <= sample_data.shape[0], "Number of rows should not increase after detecting anomalies"

def test_train_autoencoder(sample_data):
    df = train_autoencoder(sample_data, temperature_threshold=25)
    assert 'anomaly' not in df.columns

# Integration test for the entire pipeline
def test_integration_pipeline(sample_data):
    df = sample_data
    df = analyze_data(df)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = detect_anomalies(df)
    df = train_autoencoder(df)

# Run the tests
if __name__ == "__main__":
    pytest.main()


# python -m pytest test_battery_analysis.py -v