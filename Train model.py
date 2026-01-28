"""
Setup and Model Training Script
Run this script to train both models and prepare for deployment
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import DataPreprocessor
from forecasting_utils import TimeSeriesForecaster, ForecastingMetrics

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    mean_absolute_error, r2_score
)


def setup_directories():
    """Create necessary directories"""
    directories = ['../data', '../models', '../logs']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory ready: {directory}")


def load_and_prepare_data(data_path):
    """Load and prepare data"""
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)
    
    if df is None:
        print("✗ Failed to load data")
        return None, None, None
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target variable distribution:\n{df['noshow'].value_counts()}")
    
    return df, preprocessor


def train_classification_models(X_train, X_test, y_train, y_test, class_weights):
    """Train and evaluate classification models"""
    print("\n" + "="*60)
    print("STEP 2: NO-SHOW CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight=class_weights),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                                class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
                                random_state=42, verbosity=0)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall
        }
        
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
    
    # Select best model (by ROC-AUC)
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    print(f"\n✓ Best Classification Model: {best_model_name}")
    print(f"  ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
    
    return best_model, results


def train_forecasting_models(X_train, X_test, y_train, y_test):
    """Train and evaluate forecasting models"""
    print("\n" + "="*60)
    print("STEP 3: DEMAND FORECASTING MODEL TRAINING")
    print("="*60)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, 
                               random_state=42, verbosity=0)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_absolute_error(y_test, y_pred) ** 2)
        mape = TimeSeriesForecaster.calculate_mape(y_test.values, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
        
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
    
    # Select best model (by MAPE)
    best_model_name = min(results.keys(), key=lambda x: results[x]['mape'])
    best_model = results[best_model_name]['model']
    
    print(f"\n✓ Best Forecasting Model: {best_model_name}")
    print(f"  MAPE: {results[best_model_name]['mape']:.2f}%")
    
    return best_model, results


def save_models(classification_model, forecasting_model, preprocessor, output_dir='../models'):
    """Save trained models"""
    print("\n" + "="*60)
    print("STEP 4: SAVING MODELS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save classification model
    clf_path = os.path.join(output_dir, 'noshow_classifier.pkl')
    joblib.dump(classification_model, clf_path)
    print(f"✓ Classification model saved: {clf_path}")
    
    # Save forecasting model
    for_path = os.path.join(output_dir, 'demand_forecast_model.pkl')
    joblib.dump(forecasting_model, for_path)
    print(f"✓ Forecasting model saved: {for_path}")
    
    # Save preprocessor
    pre_path = os.path.join(output_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, pre_path)
    print(f"✓ Preprocessor saved: {pre_path}")
    
    print(f"\n✓ All models saved successfully in {output_dir}/")


def create_summary_report(clf_results, for_results):
    """Create and display summary report"""
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY REPORT")
    print("="*60)
    
    print("\nCLASSIFICATION MODELS:")
    print("-" * 60)
    for model_name, metrics in clf_results.items():
        print(f"\n{model_name}:")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
    
    print("\n\nFORECASTING MODELS:")
    print("-" * 60)
    for model_name, metrics in for_results.items():
        print(f"\n{model_name}:")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  R²:   {metrics['r2']:.4f}")
    
    print("\n" + "="*60)


def main():
    """Main training pipeline"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "MEDICAL APPOINTMENT NO-SHOW PREDICTION" + " "*10 + "║")
    print("║" + " "*15 + "Model Training Pipeline" + " "*21 + "║")
    print("╚" + "="*58 + "╝")
    
    # Setup
    setup_directories()
    
    # Load data
    data_path = '../data/medical_appointments.csv'
    
    if not os.path.exists(data_path):
        print(f"\n✗ Data file not found at {data_path}")
        print("Please download the dataset and place it in the data/ folder")
        return
    
    df, preprocessor = load_and_prepare_data(data_path)
    
    if df is None:
        return
    
    # Prepare classification data
    X_clf, y_clf, class_weights = preprocessor.prepare_classification_data(df)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, stratify=y_clf, random_state=42
    )
    
    # Train classification models
    best_clf_model, clf_results = train_classification_models(
        X_train_clf, X_test_clf, y_train_clf, y_test_clf, class_weights
    )
    
    # Prepare time series data
    ts_data = preprocessor.prepare_time_series_data(df)
    ts_data_lag = TimeSeriesForecaster.create_lag_features(ts_data['daily_count'].values)
    
    # Add temporal features
    temporal_features = TimeSeriesForecaster.add_temporal_features(
        ts_data['appointment_date'].iloc[7:].values
    )
    
    X_ts = pd.concat([ts_data_lag.reset_index(drop=True), 
                      temporal_features.reset_index(drop=True)], axis=1)
    y_ts = ts_data['daily_count'].iloc[7:].values
    
    # Chronological split for time series
    split_idx = int(len(X_ts) * 0.8)
    X_train_ts = X_ts.iloc[:split_idx]
    X_test_ts = X_ts.iloc[split_idx:]
    y_train_ts = y_ts[:split_idx]
    y_test_ts = y_ts[split_idx:]
    
    # Train forecasting models
    best_for_model, for_results = train_forecasting_models(
        X_train_ts, X_test_ts, y_train_ts, y_test_ts
    )
    
    # Save models
    save_models(best_clf_model, best_for_model, preprocessor)
    
    # Create report
    create_summary_report(clf_results, for_results)
    
    print("\n✓ Training pipeline completed successfully!")
    print("\nNext steps:")
    print("  1. Navigate to the app/ folder")
    print("  2. Run: streamlit run app.py")
    print("  3. Open browser at http://localhost:8501")
    

if __name__ == "__main__":
    main()
