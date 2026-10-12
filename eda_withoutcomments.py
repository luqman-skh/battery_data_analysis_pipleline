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
    table = pq.read_table(filepath)
    return table.to_pandas()

def analyze_data(df):
    print("Basic Info:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
    df = df.drop_duplicates()
    return df

def handle_missing_values(df):
    df.fillna(df.median(), inplace=True)
    return df

def remove_outliers(df, threshold=3):
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = numeric_df.apply(zscore)
    return df[(np.abs(z_scores) < threshold).all(axis=1)]

def detect_anomalies(df, contamination=0.01):
    numeric_df = df.select_dtypes(include=[np.number])
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df['Anomaly'] = iso_forest.fit_predict(numeric_df)
    df = df[df['Anomaly'] == 1]
    df.drop(columns=['Anomaly'], inplace=True)
    return df

def plot_distributions(df):
    df.hist(bins=50, figsize=(12, 10))
    plt.savefig('histograms.png')
    plt.close()
    return df

def train_autoencoder(df, sample_size=250000, epochs=5, batch_size=512, temperature_threshold=33):
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    X = df_sample.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(8, activation="relu")(input_layer)
    decoded = Dense(input_dim, activation="linear")(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

    X_full = scaler.transform(df.values)
    reconstructed = autoencoder.predict(X_full, batch_size=batch_size)
    mse = np.mean(np.power(X_full - reconstructed, 2), axis=1)

    threshold = np.percentile(mse, 95)
    df["anomaly"] = mse < threshold
    df = df[df["anomaly"]]
    df = df[df['temperature'] >= temperature_threshold]
    df.drop(columns=["anomaly"], inplace=True)

    return df

def final_plot(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['discharge_capacity'], df['voltage'], alpha=0.5)
    plt.title('Scatter Plot: discharge_capacity vs voltage')
    plt.xlabel('discharge_capacity')
    plt.ylabel('voltage')
    plt.savefig('final_scatter.png')
    plt.close()
    return df