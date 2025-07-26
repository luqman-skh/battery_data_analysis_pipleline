from metaflow import FlowSpec, step, Parameter
import numpy as np
import mlflow
from eda import load_data, analyze_data, handle_missing_values, remove_outliers, detect_anomalies, plot_distributions, train_autoencoder, final_plot
from model import train_predict_model

class BatteryLifeFlow(FlowSpec):
    filepath = Parameter('filepath', 
                       default='case_study_sample_dataset',
                       help='Path to the input parquet file')
    
    mlflow_tracking_uri = Parameter('mlflow_uri',
                                   default='file:///app/mlruns',
                                   help='MLflow tracking server URI')

    @step
    def start(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment("Battery_Life_Prediction")
        
        self.df = load_data(self.filepath)
        self.next(self.eda)

    @step
    def eda(self):
        self.df = analyze_data(self.df)
        self.df = self.df.select_dtypes(include=[np.number])
        self.df = plot_distributions(self.df)
        self.next(self.preprocess)

    @step
    def preprocess(self):
        self.df = handle_missing_values(self.df)
        self.df = remove_outliers(self.df)
        self.df = detect_anomalies(self.df)
        self.df = train_autoencoder(self.df)
        self.df = final_plot(self.df)
        self.next(self.train_model)

    @step
    def train_model(self):
        """Train model and track with MLflow"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment("Battery_Life_Prediction")
        self.model, self.mse = train_predict_model(self.df)
        print(f"Model trained with MSE: {self.mse}")
        self.next(self.end)

    @step
    def end(self):
        print("Flow completed successfully! Model logged to MLflow")

if __name__ == '__main__':
    BatteryLifeFlow()