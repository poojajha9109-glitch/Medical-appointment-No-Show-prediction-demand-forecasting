# Medical Appointment No-Show Prediction & Demand Forecasting

A comprehensive machine learning solution for predicting patient no-show appointments and forecasting daily appointment demand for a rehabilitation clinic. Built with Python, scikit-learn, XGBoost, and Streamlit.

## Project Overview

This project addresses critical operational challenges at a rehabilitation facility:
- **31.8% no-show rate** (vs. typical 10-20%)
- Unpredictable daily appointment demand
- Inefficient staff scheduling across specialties and locations
- Revenue loss from missed appointments

## Key Features

### 1. **No-Show Prediction (Binary Classification)**
- Predicts appointment attendance risk for individual patients
- Input: Patient demographics, health conditions, appointment details, weather
- Output: Risk probability score (0-1) and risk classification
- **Target Metrics:** F1-Score ≥ 0.70, ROC-AUC ≥ 0.75

### 2. **Demand Forecasting (Time Series Regression)**
- Predicts daily appointment volumes
- Supports overall and specialty-specific forecasting
- Captures temporal patterns, seasonality, and weather effects
- **Target Metrics:** MAPE ≤ 20%, R² ≥ 0.65

### 3. **Interactive Streamlit Dashboard**
- User-friendly interface for clinic staff
- Real-time predictions with instant feedback
- Visual analytics and insights
- Input validation and error handling

## Project Structure

```
medical-appointment-no-show-forecasting/
│
├── data/
│   └── medical_appointments.csv          # Dataset (109,593 rows, 26 columns)
│
├── notebooks/
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis
│   ├── 02_preprocessing_features.ipynb   # Data Cleaning & Feature Engineering
│   ├── 03_noshow_modeling.ipynb          # Classification Model Development
│   └── 04_demand_forecasting.ipynb       # Time Series Forecasting
│
├── models/
│   ├── noshow_classifier.pkl             # Trained classification model
│   ├── demand_forecast_model.pkl         # Trained regression model
│   └── preprocessor.pkl                  # Feature preprocessor
│
├── app/
│   ├── app.py                            # Main Streamlit application
│   ├── preprocessing.py                  # Preprocessing functions
│   ├── forecasting_utils.py              # Time series utilities
│   └── visualization.py                  # Plotting functions
│
├── requirements.txt                       # Python dependencies
├── .gitignore                            # Git ignore file
└── README.md                             # This file
```

## Dataset Information

**Source:** Medical Appointments Dataset  
**Size:** 109,593 appointments, 26 features  
**Target Variable:** `noshow` (Yes/No)

### Key Features:
- **Patient Info:** Gender, Age, Under 12, Over 60, Disability, Needs Companion
- **Appointment Details:** Specialty, Time, Shift, Date, Location (13 cities)
- **Health Conditions:** Hypertension, Diabetes, Alcoholism, Handicap, Scholarship
- **Environmental:** Temperature, Rainfall, Heat Intensity, Rain Intensity, Storm Day Before
- **Engagement:** SMS Received, Appointment Days Advance

### Missing Values Handling:
- Age: 21 missing → median imputation
- Specialty: 18 missing → "Unknown" category
- Disability: 15 missing → mode imputation
- Place: 10.5 missing → "Unknown" category
- Weather: ~2% missing → interpolation

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/medical-appointment-no-show-forecasting.git
cd medical-appointment-no-show-forecasting
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Download the medical appointments CSV from the project resources and place it in the `data/` folder.

## Running the Project

### Option A: Run Streamlit App (Recommended)
```bash
cd app
streamlit run app.py
```
The app opens at `http://localhost:8501`

### Option B: Run Jupyter Notebooks
```bash
jupyter notebook notebooks/
```
Execute notebooks in order:
1. `01_eda.ipynb` - Explore data patterns
2. `02_preprocessing_features.ipynb` - Prepare features
3. `03_noshow_modeling.ipynb` - Train classification models
4. `04_demand_forecasting.ipynb` - Train forecasting models

## Model Development Summary

### No-Show Classification

**Algorithms Compared:**
1. Logistic Regression (baseline)
2. Random Forest Classifier
3. XGBoost Classifier

**Class Imbalance Handling:**
- Stratified train-test split
- Class weight balancing
- Optional SMOTE for synthetic samples
- Appropriate evaluation metrics (F1, ROC-AUC, Precision-Recall)

**Feature Importance:**
Top predictors typically include:
- Days advance notice
- SMS received
- Age and age flags
- Hypertension and chronic conditions
- Weather conditions

### Demand Forecasting

**Algorithms Compared:**
1. Naive Baseline (yesterday's value)
2. ARIMA/SARIMA (classical time series)
3. Random Forest with lag features
4. XGBoost Time Series

**Feature Engineering:**
- Lag features (t-1, t-2, t-7, t-30)
- Day of week and month
- Rolling averages
- Seasonal indicators
- Weather variables

**Evaluation:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

## Streamlit App Features

### Tab 1: No-Show Risk Predictor
- Input patient demographics (age, gender, location)
- Select appointment details (specialty, date, shift)
- Check health conditions
- **Output:** Risk score with visual gauge and recommendations

### Tab 2: Demand Forecaster
- Select forecast period (7, 14, 30 days)
- Choose specialty filter (optional)
- **Output:** Demand forecast chart with confidence intervals

### Additional Features:
- Model performance metrics dashboard
- Feature importance visualizations
- Data distribution insights
- Load time monitoring (<2 seconds target)

## Business Use Cases

1. **Risk-Based Patient Engagement**
   - Identify high-risk patients
   - Trigger SMS reminders or alternative scheduling

2. **Intelligent Staffing**
   - Optimize specialist scheduling based on demand forecast
   - Reduce overstaffing/understaffing

3. **Revenue Protection**
   - Reduce no-show losses through early intervention
   - Estimate impact: ~5-10% no-show reduction

4. **Geographic Resource Planning**
   - Allocate specialists across 13 cities
   - Optimize equipment distribution

5. **Specialty-Level Planning**
   - Forecast demand by specialty type
   - Balance workload across services

6. **Seasonal Adaptation**
   - Account for weather and seasonal patterns
   - Proactive capacity planning

## Key Metrics & Performance

### Classification Model Target
- **F1-Score:** ≥ 0.70
- **ROC-AUC:** ≥ 0.75
- **Precision & Recall:** Balanced for business needs

### Forecasting Model Target
- **MAPE:** ≤ 20%
- **R² Score:** ≥ 0.65
- **Prediction Latency:** < 2 seconds

## Technical Stack

**Data Processing:**
- Pandas, NumPy
- scikit-learn preprocessing

**Machine Learning:**
- scikit-learn (Logistic Regression, Random Forest)
- XGBoost, LightGBM
- statsmodels (ARIMA/SARIMA)

**Time Series:**
- Prophet (optional for advanced forecasting)
- Custom lag-based features

**Visualization:**
- Matplotlib, Seaborn
- Plotly (interactive charts)
- Streamlit components

**Deployment:**
- Streamlit (web framework)
- Joblib (model serialization)

## File Descriptions

### notebooks/01_eda.ipynb
Exploratory Data Analysis covering:
- Data loading and overview
- Missing value analysis
- Distribution analysis (numerical and categorical)
- Correlation analysis
- No-show rate by patient groups
- Temporal patterns
- Geographic analysis

### notebooks/02_preprocessing_features.ipynb
Data preparation including:
- Missing value imputation strategies
- Outlier detection and handling
- Categorical encoding (OneHot, Target encoding)
- Numerical feature scaling
- Temporal feature engineering
- Feature interaction creation
- Pipeline definition

### notebooks/03_noshow_modeling.ipynb
Classification model development:
- Train-test split with stratification
- Baseline model (Logistic Regression)
- Advanced models (Random Forest, XGBoost)
- Hyperparameter tuning
- Cross-validation
- Model evaluation and comparison
- Feature importance analysis
- Model serialization

### notebooks/04_demand_forecasting.ipynb
Time series forecasting:
- Data aggregation to daily level
- Trend and seasonality analysis
- Lag feature engineering
- Train-test split (chronological)
- Baseline model comparison
- Advanced models implementation
- Evaluation metrics
- Specialty-level forecasting

### app/app.py
Main Streamlit application with:
- Model loading and caching
- Two-tab interface
- Patient input form
- Prediction display
- Demand forecasting interface
- Performance metrics
- Error handling

### app/preprocessing.py
Reusable preprocessing functions:
- Pipeline building
- Data validation
- Feature scaling
- Categorical encoding

### app/forecasting_utils.py
Time series utilities:
- Lag feature generation
- Temporal feature engineering
- Forecast aggregation
- Confidence interval calculation

### app/visualization.py
Visualization components:
- Risk gauge charts
- Time series plots
- Distribution charts
- Performance metrics dashboards

## Model Deployment

### Local Deployment (Development)
```bash
streamlit run app.py
```

### Production Considerations
- Use environment variables for sensitive data
- Implement logging and monitoring
- Set up automated retraining pipeline
- Create prediction logging system
- Monitor model performance drift

## Troubleshooting

### Common Issues

**Issue:** Model files not found
```bash
# Ensure models are trained and saved
python notebooks/03_noshow_modeling.ipynb
python notebooks/04_demand_forecasting.ipynb
```

**Issue:** Streamlit cache errors
```bash
streamlit cache clear
streamlit run app.py
```

**Issue:** Missing dependencies
```bash
pip install -r requirements.txt --upgrade
```

## Future Enhancements

1. **Advanced Models**
   - Deep learning (LSTM for time series)
   - Ensemble methods (stacking, blending)
   - AutoML frameworks

2. **Feature Expansion**
   - Social media sentiment analysis
   - Traffic/commute time data
   - Holiday calendar integration
   - Patient interaction history

3. **Deployment**
   - Docker containerization
   - Cloud deployment (AWS, GCP, Azure)
   - REST API endpoints
   - Real-time predictions with message queues

4. **Monitoring**
   - Model performance tracking
   - Prediction explanation (SHAP values)
   - A/B testing framework
   - Automated retraining

## References

- Streamlit Documentation: https://docs.streamlit.io
- Scikit-learn: https://scikit-learn.org
- XGBoost: https://xgboost.readthedocs.io
- Time Series Forecasting: https://statsmodels.org

## Project Guidelines

✅ Use chronological train-test split for demand forecasting  
✅ Handle class imbalance (31.8% no-show) with weighted loss  
✅ Save models with joblib for reproducibility  
✅ Create responsive Streamlit layouts with tabs  
✅ Include input validation and error handling  
✅ Add loading spinners for predictions  
✅ Document all preprocessing steps  
✅ Compare minimum 3 algorithms per task  

## Timeline

- **Days 1-2:** EDA and preprocessing
- **Days 3-4:** No-show model development
- **Days 5-6:** Demand forecasting model
- **Days 7-8:** Streamlit app development
- **Days 9-10:** Testing, optimization, and deployment


