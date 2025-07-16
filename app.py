import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

# =============================================
# APP CONFIGURATION
# =============================================
st.set_page_config(
    page_title="Air Quality Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌤️"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --good: #2ecc71;
        --fair: #f39c12;
        --poor: #e74c3c;
        --primary: #3498db;
    }
    
    .good-card {
        background-color: #e8f5e9;
        border-left: 5px solid var(--good);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    .fair-card {
        background-color: #fff8e1;
        border-left: 5px solid var(--fair);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    .poor-card {
        background-color: #ffebee;
        border-left: 5px solid var(--poor);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    .metric-box {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# DATA & MODEL FUNCTIONS
# =============================================
@st.cache_data
def load_data():
    """Load air quality data"""
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.csv"
        df = pd.read_csv(url, sep=';', decimal=',', engine='python')
        return df.iloc[:, :-2]  # Remove empty columns
    except:
        try:
            local_path = os.path.join(os.path.dirname(__file__), "AirQualityUCI.csv")
            return pd.read_csv(local_path, sep=';', decimal=',', engine='python').iloc[:, :-2]
        except:
            return None

def clean_data(df, threshold=-200):
    """Clean and prepare data"""
    df = df.copy()
    df.replace(to_replace=range(-1000, int(threshold+1)), value=np.nan, inplace=True)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                  format='%d/%m/%Y %H.%M.%S', errors='coerce')
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.set_index('datetime', inplace=True)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '_').str.replace(')', '')
    return df.dropna()

def engineer_features(df, threshold_std=1.5, window=3):
    """Create features for modeling"""
    target_col = 'PT08.S5_O3'
    sensors = ['PT08.S1_CO', 'PT08.S2_NMHC', 'PT08.S3_NOx', 'PT08.S4_NO2', target_col]
    
    pct_change = df[target_col].pct_change(periods=window).shift(-window)
    threshold = threshold_std * pct_change.std()
    df['target'] = np.where(pct_change < -threshold, 1, np.where(pct_change > threshold, -1, 0))
    
    for sensor in sensors:
        for lag in [1, 2, 3, 6, 12]:
            df[f'{sensor}_lag{lag}'] = df[sensor].shift(lag)
        df[f'{sensor}_roll3'] = df[sensor].rolling(3).mean()
        df[f'{sensor}_diff'] = df[sensor].diff()
    
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    return df.dropna()

def train_models(X_train, y_train, X_test, y_test, rf_params=None, gb_params=None):
    """Train and evaluate models"""
    # Initialize parameters if None
    rf_params = rf_params or {}
    gb_params = gb_params or {}
    
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_params.get('n_estimators', 200),
            max_depth=rf_params.get('max_depth', 10),
            class_weight='balanced',
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=gb_params.get('n_estimators', 100),
            learning_rate=gb_params.get('learning_rate', 0.05),
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    }
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    for name, model in models.items():
        try:
            min_samples = y_train.value_counts().min()
            if min_samples > 5:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))
                X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
            else:
                X_res, y_res = X_train_scaled, y_train
            
            model.fit(X_res, y_res)
            y_pred = model.predict(X_test_scaled)
            
            results[name] = {
                "model": model,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred, average='weighted'),
                "scaler": scaler
            }
        except Exception as e:
            st.error(f"Error training {name}: {str(e)}")
    
    return results

# =============================================
# INITIALIZATION & SESSION STATE
# =============================================
def initialize_app():
    """Initialize session state"""
    defaults = {
        "data_loaded": False,
        "models_trained": False,
        "current_model": None,
        "best_model": None,
        "threshold_std": 1.5,
        "prediction_window": 3,
        "cleaning_threshold": -200,
        "model_performance": {},
        "initial_training_done": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_app()

# =============================================
# AUTOMATIC INITIAL TRAINING
# =============================================
if not st.session_state.initial_training_done:
    with st.spinner("Setting up air quality models..."):
        try:
            raw_df = load_data()
            if raw_df is not None:
                cleaned_df = clean_data(raw_df, st.session_state.cleaning_threshold)
                engineered_df = engineer_features(cleaned_df, st.session_state.threshold_std, st.session_state.prediction_window)
                
                features = [col for col in engineered_df.columns 
                          if col.startswith('PT08') or col in ['hour', 'dayofweek', 'is_weekend']]
                X = engineered_df[features]
                y = engineered_df['target']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                model_results = train_models(X_train, y_train, X_test, y_test)
                
                st.session_state.update({
                    "data_loaded": True,
                    "models_trained": True,
                    "engineered_df": engineered_df,
                    "training_features": features,
                    "model_performance": model_results,
                    "best_model": max(model_results.items(), key=lambda x: x[1]['f1_score'])[0],
                    "current_model": max(model_results.items(), key=lambda x: x[1]['f1_score'])[0],
                    "initial_training_done": True
                })
        except Exception as e:
            st.error(f"Initial setup failed: {str(e)}")

# =============================================
# MAIN APP INTERFACE
# =============================================
st.title("🌤️ Air Quality Monitoring and Prediction System")

# Tab selection
user_tab, dev_tab = st.tabs(["For Everyone", "For Developers"])

# USER-FACING TAB
with user_tab:
    st.header("Check Air Quality Now")
    st.markdown("Enter current conditions to get air quality status.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_o3 = st.slider("Ozone Level (ppb)", 500, 2000, 1000)
        co_level = st.slider("CO Level (ppb)", 500, 2000, 1000)
        nox_level = st.slider("NOx Level (ppb)", 500, 2000, 1000)
    
    with col2:
        hour = st.slider("Hour of Day", 0, 23, 12)
        is_weekend = st.selectbox("Is Weekend?", ["No", "Yes"])
        o3_lag1 = st.slider("Ozone 1 Hour Ago (ppb)", 500, 2000, 950)
    
    if st.button("Get Air Quality Status", type="primary", use_container_width=True):
        if not st.session_state.models_trained:
            st.warning("System is initializing. Please wait...")
        else:
            with st.spinner("Analyzing conditions..."):
                try:
                    input_data = {
                        'PT08.S1_CO': co_level,
                        'PT08.S3_NOx': nox_level,
                        'PT08.S5_O3': current_o3,
                        'PT08.S5_O3_lag1': o3_lag1,
                        'hour': hour,
                        'is_weekend': 1 if is_weekend == "Yes" else 0
                    }
                    
                    features = st.session_state.training_features
                    for feature in features:
                        if feature not in input_data:
                            input_data[feature] = st.session_state.engineered_df[feature].median()
                    
                    input_df = pd.DataFrame([input_data])[features]
                    model_info = st.session_state.model_performance[st.session_state.current_model]
                    scaled_input = model_info['scaler'].transform(input_df)
                    prediction = model_info['model'].predict(scaled_input)[0]
                    
                    if prediction == -1:
                        st.markdown("""
                        <div class="poor-card">
                            <h3>🚨 Poor Air Quality</h3>
                            <p>High ozone levels detected. Recommendations:</p>
                            <ul>
                                <li>Reduce outdoor activities</li>
                                <li>Close windows</li>
                                <li>Monitor symptoms if sensitive</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    elif prediction == 1:
                        st.markdown("""
                        <div class="fair-card">
                            <h3>⚠️ Improving Air Quality</h3>
                            <p>Ozone levels are dropping. Recommendations:</p>
                            <ul>
                                <li>Conditions should improve soon</li>
                                <li>Still be cautious if sensitive</li>
                                <li>Check again in 1-2 hours</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="good-card">
                            <h3>✅ Good Air Quality</h3>
                            <p>Ozone levels are normal. Recommendations:</p>
                            <ul>
                                <li>Safe for outdoor activities</li>
                                <li>No special precautions needed</li>
                                <li>Enjoy your day!</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error("Couldn't complete prediction. Please try again.")

# DEVELOPER TAB
with dev_tab:
    st.header("Model Development Console")
    
    with st.expander("Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Settings")
            new_clean_thresh = st.slider("Data cleaning threshold", -300, 0, st.session_state.cleaning_threshold)
            new_threshold_std = st.slider("Change threshold (std dev)", 0.5, 3.0, st.session_state.threshold_std, 0.1)
            new_window = st.selectbox("Prediction window (hours)", [1, 2, 3, 6, 12], 
                                    index=[1, 2, 3, 6, 12].index(st.session_state.prediction_window))
        
        with col2:
            st.subheader("Model Selection")
            if st.session_state.models_trained:
                model_options = list(st.session_state.model_performance.keys())
                current_idx = model_options.index(st.session_state.current_model) if st.session_state.current_model else 0
                selected_model = st.selectbox("Active model", model_options, index=current_idx)
                st.session_state.current_model = selected_model
    
    if st.button("Retrain Models", type="primary"):
        with st.spinner("Retraining models..."):
            try:
                raw_df = load_data()
                if raw_df is not None:
                    cleaned_df = clean_data(raw_df, new_clean_thresh)
                    engineered_df = engineer_features(cleaned_df, new_threshold_std, new_window)
                    
                    features = [col for col in engineered_df.columns 
                              if col.startswith('PT08') or col in ['hour', 'dayofweek', 'is_weekend']]
                    X = engineered_df[features]
                    y = engineered_df['target']
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    model_results = train_models(X_train, y_train, X_test, y_test)
                    
                    st.session_state.update({
                        "data_loaded": True,
                        "models_trained": True,
                        "engineered_df": engineered_df,
                        "training_features": features,
                        "model_performance": model_results,
                        "cleaning_threshold": new_clean_thresh,
                        "threshold_std": new_threshold_std,
                        "prediction_window": new_window,
                        "best_model": max(model_results.items(), key=lambda x: x[1]['f1_score'])[0]
                    })
                    
                    st.success("Models updated successfully!")
            except Exception as e:
                st.error(f"Error retraining: {str(e)}")

    if st.session_state.models_trained:
        st.subheader("Model Performance")
        cols = st.columns(len(st.session_state.model_performance))
        
        for (name, metrics), col in zip(st.session_state.model_performance.items(), cols):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <h4>{name}</h4>
                    <p>Accuracy: {metrics['accuracy']:.2f}</p>
                    <p>F1 Score: {metrics['f1_score']:.2f}</p>
                    {"🌟" if name == st.session_state.best_model else ""}
                </div>
                """, unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.title("System Status")
    
    if st.session_state.data_loaded:
        st.markdown("""
        <div class="metric-box">
            <h4>Data Loaded</h4>
            <p>Records: {:,}</p>
            <p>Key Features: 6</p>
            <p>Trained and Engineered Features: {}</p>
        </div>
        """.format(
            len(st.session_state.engineered_df),
            len(st.session_state.training_features)
        ), unsafe_allow_html=True)
    else:
        st.warning("Data not loaded")
    
    if st.session_state.models_trained:
        st.markdown("""
        <div class="metric-box">
            <h4>Active Model</h4>
            <p>{}</p>
            <p>Accuracy: {:.2f}</p>
        </div>
        """.format(
            st.session_state.current_model,
            st.session_state.model_performance[st.session_state.current_model]['accuracy']
        ), unsafe_allow_html=True)
    else:
        st.warning("Models not trained")
    
    if st.button("Refresh System"):
        st.session_state.initial_training_done = False
        st.rerun()