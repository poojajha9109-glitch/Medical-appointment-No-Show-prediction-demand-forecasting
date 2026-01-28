""
Data Preprocessing and Feature Engineering Utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime

class DataPreprocessor:
    """Handles all data preprocessing and feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.preprocessor = None
        self.feature_names = []
    
    def load_data(self, filepath):
        """Load data from CSV"""
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def handle_missing_values(self, df):
        """Handle missing values strategically"""
        df = df.copy()
        
        # Define strategies for each column
        missing_strategies = {
            'age': 'median',
            'specialty': 'mode',
            'disability': 'mode',
            'place': 'mode',
            'avg_temp': 'mean',
            'rainfall': 'mean',
            'rain_intensity': 'mean',
            'heat_intensity': 'mean'
        }
        
        for col, strategy in missing_strategies.items():
            if col in df.columns:
                if df[col].isnull().sum() > 0:
                    if strategy == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
                    elif strategy == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif strategy == 'mode':
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        print(f"✓ Missing values handled")
        return df
    
    def create_temporal_features(self, df, date_column='appointment_date'):
        """Create temporal features from date column"""
        df = df.copy()
        
        if date_column not in df.columns:
            print(f"✗ {date_column} not found in dataframe")
            return df
        
        # Convert to datetime if not already
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Extract temporal features
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['day_of_month'] = df[date_column].dt.day
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['week_of_year'] = df[date_column].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
        
        print(f"✓ Temporal features created")
        return df
    
    def create_age_features(self, df, age_column='age'):
        """Create age-based features"""
        df = df.copy()
        
        if age_column not in df.columns:
            print(f"✗ {age_column} not found in dataframe")
            return df
        
        # Age categories
        df['is_child'] = (df[age_column] < 12).astype(int)
        df['is_senior'] = (df[age_column] > 60).astype(int)
        df['age_group'] = pd.cut(df[age_column], 
                                  bins=[0, 12, 18, 35, 60, 120],
                                  labels=['Child', 'Teen', 'Adult', 'Senior', 'Elderly'])
        
        print(f"✓ Age features created")
        return df
    
    def create_weather_features(self, df):
        """Create weather-based features"""
        df = df.copy()
        
        weather_cols = ['avg_temp', 'rainfall', 'rain_intensity', 'heat_intensity']
        
        # Create weather severity indicator
        if 'rainfall' in df.columns:
            df['is_rainy'] = (df['rainfall'] > 0).astype(int)
        
        if 'rain_intensity' in df.columns:
            df['heavy_rain'] = (df['rain_intensity'] > 0.7).astype(int)
        
        if 'avg_temp' in df.columns:
            df['extreme_cold'] = (df['avg_temp'] < 5).astype(int)
            df['extreme_heat'] = (df['avg_temp'] > 40).astype(int)
        
        # Weather composite indicator
        condition_cols = [col for col in ['is_rainy', 'heavy_rain', 'extreme_cold', 'extreme_heat'] 
                         if col in df.columns]
        if condition_cols:
            df['bad_weather'] = df[condition_cols].sum(axis=1) > 0
            df['bad_weather'] = df['bad_weather'].astype(int)
        
        print(f"✓ Weather features created")
        return df
    
    def encode_categorical(self, df, categorical_cols=None):
        """Encode categorical variables"""
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            # Exclude target and date columns
            categorical_cols = [col for col in categorical_cols 
                              if col not in ['noshow', 'appointment_date']]
        
        # One-hot encoding
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, 
                                  columns=encoder.get_feature_names_out(categorical_cols))
        
        # Replace original categorical columns
        df_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
        
        print(f"✓ Categorical variables encoded")
        return df_encoded
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance for classification task"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
        
        print(f"✓ Class weights computed: {class_weight_dict}")
        return class_weight_dict
    
    def build_preprocessing_pipeline(self, numeric_features, categorical_features):
        """Build a preprocessing pipeline"""
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        self.preprocessor = preprocessor
        print(f"✓ Preprocessing pipeline created")
        return preprocessor
    
    def prepare_classification_data(self, df, target_col='noshow'):
        """Prepare data for classification task"""
        df = df.copy()
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Create features
        df = self.create_temporal_features(df)
        df = self.create_age_features(df)
        df = self.create_weather_features(df)
        
        # Step 3: Separate features and target
        X = df.drop(columns=[target_col, 'appointment_date'], errors='ignore')
        y = df[target_col].map({'No': 0, 'Yes': 1})
        
        # Step 4: Handle class imbalance
        class_weights = self.handle_class_imbalance(X, y)
        
        print(f"✓ Classification data prepared")
        return X, y, class_weights
    
    def prepare_time_series_data(self, df, date_col='appointment_date', group_col=None):
        """Prepare data for time series forecasting"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        if group_col:
            # Group by date and category
            ts_data = df.groupby([date_col, group_col]).size().reset_index(name='daily_count')
        else:
            # Group by date only
            ts_data = df.groupby(date_col).size().reset_index(name='daily_count')
        
        ts_data = ts_data.sort_values(date_col).reset_index(drop=True)
        
        print(f"✓ Time series data prepared: {len(ts_data)} records")
        return ts_data
    
    @staticmethod
    def create_lag_features(df, target_col='daily_count', lags=[1, 2, 7, 14, 30]):
        """Create lag features for time series"""
        df = df.copy()
        
        for lag in lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Add rolling averages
        df['rolling_mean_7'] = df[target_col].rolling(window=7).mean()
        df['rolling_mean_14'] = df[target_col].rolling(window=14).mean()
        df['rolling_std_7'] = df[target_col].rolling(window=7).std()
        
        # Remove rows with NaN from lag creation
        df = df.dropna()
        
        print(f"✓ Lag features created")
        return df
    
    @staticmethod
    def normalize_features(X_train, X_test):
        """Normalize numerical features"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✓ Features normalized")
        return X_train_scaled, X_test_scaled, scaler


# Helper functions
def get_feature_names(preprocessor):
    """Extract feature names from preprocessor"""
    feature_names = []
    
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(columns)
        elif name == 'cat':
            # Get one-hot encoded feature names
            if hasattr(transformer, 'named_steps'):
                onehot = transformer.named_steps['onehot']
                feature_names.extend(onehot.get_feature_names_out(columns).tolist())
    
    return feature_names


def validate_input(input_dict):
    """Validate input data"""
    required_fields = ['age', 'gender', 'specialty', 'place']
    
    for field in required_fields:
        if field not in input_dict:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate age
    if not (0 <= input_dict['age'] <= 120):
        raise ValueError("Age must be between 0 and 120")
    
    return True


def prepare_single_prediction(input_dict, preprocessor):
    """Prepare a single sample for prediction"""
    input_df = pd.DataFrame([input_dict])
    input_processed = preprocessor.transform(input_df)
    return input_processed
