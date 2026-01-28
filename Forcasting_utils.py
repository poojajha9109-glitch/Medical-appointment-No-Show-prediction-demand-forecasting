"
Time Series Forecasting Utilities
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TimeSeriesForecaster:
    """Utilities for time series forecasting and evaluation"""
    
    @staticmethod
    def calculate_mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100
        return mape
    
    @staticmethod
    def create_lag_features(series, lags=[1, 2, 7, 14, 30]):
        """Create lag features from time series"""
        df = pd.DataFrame({'value': series})
        
        for lag in lags:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Rolling averages
        df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
        df['rolling_mean_14'] = df['value'].rolling(window=14).mean()
        df['rolling_std_7'] = df['value'].rolling(window=7).std()
        
        # Exponential weighted moving average
        df['ewma_7'] = df['value'].ewm(span=7).mean()
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    @staticmethod
    def add_temporal_features(dates):
        """Add temporal features to dates"""
        df = pd.DataFrame({'date': dates})
        df['date'] = pd.to_datetime(df['date'])
        
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
        
        return df.drop('date', axis=1)
    
    @staticmethod
    def train_test_split_ts(X, y, test_size=0.2):
        """Chronological train-test split for time series"""
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def evaluate_forecast(y_true, y_pred):
        """Evaluate forecasting model"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = TimeSeriesForecaster.calculate_mape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        return metrics
    
    @staticmethod
    def generate_forecast_with_ci(model, last_values, future_periods=30, 
                                   confidence=0.95, lag_order=7):
        """Generate forecast with confidence intervals"""
        predictions = []
        forecast_values = list(last_values)
        
        for _ in range(future_periods):
            # Use last lag_order values for prediction
            X_last = np.array(forecast_values[-lag_order:]).reshape(1, -1)
            pred = model.predict(X_last)[0]
            predictions.append(pred)
            forecast_values.append(pred)
        
        # Calculate confidence intervals using residual std
        pred_std = np.std(predictions) * 0.15  # 15% uncertainty margin
        
        lower_bound = np.array(predictions) - (1.96 * pred_std)
        upper_bound = np.array(predictions) + (1.96 * pred_std)
        
        # Ensure positive values
        lower_bound = np.maximum(lower_bound, 0)
        
        return np.array(predictions), lower_bound, upper_bound
    
    @staticmethod
    def aggregate_forecast_by_specialty(forecast_df, specialty_column='specialty'):
        """Aggregate forecasts by specialty"""
        if specialty_column not in forecast_df.columns:
            return forecast_df
        
        aggregated = forecast_df.groupby([forecast_df.index, specialty_column]).agg({
            'predicted_demand': 'mean',
            'lower_bound': 'mean',
            'upper_bound': 'mean'
        }).reset_index()
        
        return aggregated
    
    @staticmethod
    def calculate_seasonality_factor(historical_data, period=365):
        """Calculate seasonality factor for each day"""
        seasonality = {}
        
        for day in range(period):
            day_values = historical_data[day::period]
            if len(day_values) > 0:
                seasonality[day] = day_values.mean()
        
        return seasonality
    
    @staticmethod
    def apply_seasonality(forecast_values, seasonality_factor, period=365):
        """Apply seasonality to forecast values"""
        adjusted_forecast = []
        
        for i, pred in enumerate(forecast_values):
            day_of_year = i % period
            if day_of_year in seasonality_factor:
                adjustment = seasonality_factor[day_of_year]
                adjusted_pred = pred * (adjustment / np.mean(list(seasonality_factor.values())))
            else:
                adjusted_pred = pred
            
            adjusted_forecast.append(max(adjusted_pred, 0))
        
        return np.array(adjusted_forecast)


class ForecastingMetrics:
    """Metrics specifically for forecasting tasks"""
    
    @staticmethod
    def print_metrics(y_true, y_pred, metric_names=None):
        """Print forecasting metrics in a formatted way"""
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': TimeSeriesForecaster.calculate_mape(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        
        print("\n" + "="*50)
        print("FORECASTING METRICS")
        print("="*50)
        
        for metric_name, value in metrics.items():
            if metric_name == 'MAPE':
                print(f"{metric_name:.<30} {value:.2f}%")
            else:
                print(f"{metric_name:.<30} {value:.4f}")
        
        print("="*50 + "\n")
        
        return metrics
    
    @staticmethod
    def compare_models(model_results):
        """Compare multiple forecasting models"""
        comparison_df = pd.DataFrame(model_results)
        comparison_df = comparison_df.sort_values('MAPE')
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(comparison_df.to_string(index=False))
        print("="*70 + "\n")
        
        return comparison_df


class DemandForecaster:
    """Specialized class for demand forecasting"""
    
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.last_training_metrics = None
    
    def forecast_demand(self, historical_data, periods=30):
        """Forecast future demand"""
        last_values = historical_data[-7:].values  # Last 7 days
        
        forecast, lower_ci, upper_ci = TimeSeriesForecaster.generate_forecast_with_ci(
            self.model, last_values, future_periods=periods
        )
        
        return forecast, lower_ci, upper_ci
    
    def forecast_by_specialty(self, data, specialty_list, periods=30):
        """Forecast demand for each specialty"""
        specialty_forecasts = {}
        
        for specialty in specialty_list:
            specialty_data = data[data['specialty'] == specialty]['daily_count'].values
            
            if len(specialty_data) >= 10:  # Need minimum data
                forecast, lower, upper = self.forecast_demand(specialty_data, periods)
                specialty_forecasts[specialty] = {
                    'forecast': forecast,
                    'lower_ci': lower,
                    'upper_ci': upper
                }
        
        return specialty_forecasts
    
    def get_staffing_recommendations(self, forecast, avg_patient_per_staff=10):
        """Get staffing recommendations based on forecast"""
        avg_demand = np.mean(forecast)
        peak_demand = np.max(forecast)
        
        avg_staff = int(np.ceil(avg_demand / avg_patient_per_staff))
        peak_staff = int(np.ceil(peak_demand / (avg_patient_per_staff * 0.8)))
        
        recommendations = {
            'average_daily_appointments': int(avg_demand),
            'peak_daily_appointments': int(peak_demand),
            'recommended_avg_staff': avg_staff,
            'recommended_peak_staff': peak_staff,
            'staff_adjustment_needed': peak_staff - avg_staff
        }
        
        return recommendations


# Baseline Models for Comparison

class NaiveBaseline:
    """Simple baseline: repeat last value"""
    
    def __init__(self, lag=1):
        self.lag = lag
    
    def predict(self, X):
        """Return lagged value"""
        if hasattr(X, 'values'):
            X = X.values
        return X[-self.lag]


class SeasonalNaive:
    """Seasonal naive baseline: repeat value from same period last year"""
    
    def __init__(self, seasonal_period=365):
        self.seasonal_period = seasonal_period
    
    def predict(self, X):
        """Return seasonally lagged value"""
        if hasattr(X, 'values'):
            X = X.values
        
        if len(X) >= self.seasonal_period:
            return X[-self.seasonal_period]
        else:
            return X[-1]


# Export main classes
__all__ = [
    'TimeSeriesForecaster',
    'ForecastingMetrics',
    'DemandForecaster',
    'NaiveBaseline',
    'SeasonalNaive'
]
