import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def train_predict_model(df):
    """Train XGBoost model for discharge capacity prediction.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Processed DataFrame containing training features and target
        
    Returns
    -------
    xgboost.XGBRegressor
        Trained regression model ready for predictions
        
    Raises
    ------
    ValueError
        - If input DataFrame is empty
        - If insufficient samples for train-test split
        
    Notes
    -----
    Model Configuration:
    - Uses default XGBRegressor parameters
    - Features: ['voltage', 'temperature', 'internal_resistance', 'cycle_index']
    - Target: 'discharge_capacity'
    - 80/20 train-test split with random state 42

    Reason for selecting the model:
    - XGBoost is known for its high predictive performance. 
    - It often outperforms other algorithms in competitions and benchmarks 
    due to its ability to capture complex patterns in the data.
    - XGBoost is designed to be computationally efficient and can handle large datasets quickly. 


    
    Examples
    --------
    >>> model = train_predict_model(clean_df)
    >>> prediction = model.predict(X_new)
    """
    """Train model with MLflow tracking"""
    features = ['voltage', 'temperature', 'internal_resistance', 'cycle_index']
    target = 'discharge_capacity'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Enable auto-logging
    mlflow.xgboost.autolog()
    
    with mlflow.start_run():
        model = XGBRegressor()
        model.fit(X_train, y_train)
        
        # Calculate metrics
        predictions = model.predict(X_test)
        mse = (mean_squared_error(y_test, predictions))
        
        # Log custom metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_param("input_shape", X_train.shape)
        
        # Log model with signature
        signature = mlflow.models.infer_signature(X_train, predictions)
        mlflow.xgboost.log_model(model, "battery_model", signature=signature)
        
    return model, mse