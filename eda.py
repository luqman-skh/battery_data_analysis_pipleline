import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def load_data(filepath):
    """Load parquet dataset into a pandas DataFrame.
    
    Parameters
    ----------
    filepath : str
        Path to the parquet file to be loaded
        
    Returns
    -------
    pandas.DataFrame
        Loaded dataframe containing battery metrics
    
    Examples
    --------
    >>> df = load_data("battery_data")
    """
    table = pq.read_table(filepath)
    return table.to_pandas()

def analyze_data(df):
    """Generate basic data analysis insights and statistics.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing battery metrics data
        
    Returns
    -------
    pandas.DataFrame
        Original DataFrame unchanged
    
    Notes
    -----
    Prints following analysis:
    - Dataset shape and data types
    - Descriptive statistics for numerical columns
    - Missing values count per column
    - Number of duplicate rows
    
    Examples
    --------
    >>> df = analyze_data(raw_df)
    Basic Info:
    <class 'pandas.core.frame.DataFrame'>
    ...
    """ 
    print("Basic Info:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
    return df

def handle_missing_values(df):
    """Handle missing values using median imputation.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with potential missing values
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with missing values filled using column medians
        
    Notes
    -----
    - Modifies DataFrame in-place
    - Only affects numerical columns
    - Original column order preserved
    """
    df.fillna(df.median(), inplace=True)
    return df

def remove_outliers(df, threshold=3):
    """Remove outliers using Z-score method.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with numerical features
    threshold : float, optional
        Z-score cutoff threshold (default: 3)
        
    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with outliers removed
        
    Notes
    -----
    - Calculates Z-scores for all numerical columns
    - Removes rows where ANY column exceeds threshold
    - Uses np.number type detection for columns
    - Z-score was used as it is straightforward to calculate and interpret. 
    - It doesn't require complex algorithms or assumptions about the data distribution beyond normality.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = numeric_df.apply(zscore)
    return df[(np.abs(z_scores) < threshold).all(axis=1)]

def detect_anomalies(df, contamination=0.01):
    """Detect anomalies using Isolation Forest algorithm.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with numerical features
    contamination : float, optional
        Expected proportion of outliers in the data (default: 0.01)
        
    Returns
    -------
    pandas.DataFrame
        Dataframe with anomalous rows removed
        
    Notes
    -----
    Adds temporary 'Anomaly' column where -1 indicates anomaly,
    then filters to keep only normal data (1)
    Isolation Forest performs well in high-dimensional spaces, 
    where traditional distance-based or density-based methods may struggle
    due to the curse of dimensionality.


    """
    numeric_df = df.select_dtypes(include=[np.number])
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df['Anomaly'] = iso_forest.fit_predict(numeric_df)
    df = df[df['Anomaly'] == 1]
    df.drop(columns=['Anomaly'], inplace=True)
    return df

def plot_distributions(df):
    """Generate distribution visualizations for numerical columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing battery metrics data
        
    Returns
    -------
    pandas.DataFrame
        Original DataFrame unchanged
        
    Notes
    -----
    Creates a Histogram grid for all numerical columns
    
    Uses Matplotlib and Seaborn for visualization.
    """
    df.hist(bins=50, figsize=(12, 10))
    plt.savefig('histograms.png')
    plt.close()

    return df

def train_autoencoder(df, sample_size=250000, epochs=5, batch_size=512, temperature_threshold=33):
    """Train an autoencoder for anomaly detection with temperature filtering.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with battery metrics
    sample_size : int, optional
        Number of samples for training (default: 200000)
    epochs : int, optional
        Training epochs (default: 5)
    batch_size : int, optional
        Training batch size (default: 512)
    temperature_threshold : int, optional
        Minimum temperature to keep (default: 33)
        
    Returns
    -------
    pandas.DataFrame
        Filtered dataframe with reconstruction anomalies and
        low temperature records removed
        
    Raises
    ------
    ValueError
        If input dataframe is empty

    Notes
    -----
    Autoencoder were used as 
    Autoencoders can learn to identify anomalies without requiring labeled data. 
    This is particularly useful when labeled anomaly data is scarce or expensive to obtain.
    Autoencoders can capture complex, non-linear relationships in the data, 
    making them effective for detecting anomalies in datasets with intricate structures.
    Autoencoders can automatically learn relevant features from the data, 
    which can be more effective than manually engineered features for anomaly detection.
    """
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    X = df_sample.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(8, activation="relu")(input_layer)  # Smaller network
    decoded = Dense(input_dim, activation="linear")(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

    # Compute reconstruction error
    X_full = scaler.transform(df.values)
    reconstructed = autoencoder.predict(X_full, batch_size=batch_size)
    mse = np.mean(np.power(X_full - reconstructed, 2), axis=1)

    threshold = np.percentile(mse, 95)  # Top 5% anomalies
    df["anomaly"] = mse < threshold
    df = df[df["anomaly"]]
    df = df[df['temperature'] >= temperature_threshold]
    df.drop(columns=["anomaly"], inplace=True)

    return df

def final_plot(df):
    """Generate diagnostic scatter plot for discharge capacity vs voltage.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Processed DataFrame containing cleaned battery metrics
        
    Returns
    -------
    pandas.DataFrame
        Original DataFrame unchanged
        
    Notes
    -----
    Creates a scatter plot with:
    - discharge_capacity on x-axis
    - voltage on y-axis
    - Alpha blending for density visualization
    - Grid lines and proper labeling
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df['discharge_capacity'], df['voltage'], alpha=0.5)
    plt.title('Scatter Plot: discharge_capacity vs voltage')
    plt.xlabel('discharge_capacity')
    plt.ylabel('voltage')
    plt.savefig('final_scatter.png')
    plt.close()
    return df
