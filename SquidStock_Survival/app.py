# --- START OF FILE app.py ---

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots # Added for subplotting
import io
import base64
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, confusion_matrix
from PIL import Image, ImageDraw
import seaborn as sns # Added for heatmap (though plotly is used more now)
import matplotlib.pyplot as plt # Added for heatmap colormap (though plotly is used more now)
from io import BytesIO
import json # For prediction download
import traceback # For detailed error reporting

# Set page configuration
st.set_page_config(
    page_title="SquidStock Survival",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling and constants
NEON_PINK = "#FF0087"
NEON_GREEN = "#00FF5F"
DARK_BG = "#000000"

# Apply global custom CSS
st.markdown(f"""
<style>
/* Global Styles */
body {{
    color: white;
    background-color: black;
}}

/* Button styling with hover effects */
.stButton > button {{
    background-color: black;
    color: white;
    border: 2px solid {NEON_PINK};
    border-radius: 5px;
    padding: 0.5rem 1rem;
    transition: all 0.3s;
    width: 100%; /* Make sidebar buttons full width */
    margin-bottom: 5px; /* Add spacing between sidebar buttons */
}}

.stButton > button:hover {{
    background-color: {NEON_PINK};
    color: black;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 0, 135, 0.5);
}}

/* Specific styling for sidebar symbol buttons */
.symbol-button button {{
    font-size: 20px;
    padding: 0.3rem 0.8rem; /* Adjust padding */
    margin-bottom: 8px !important; /* More spacing */
}}

/* Selectbox styling */
.stSelectbox {{
    border-radius: 5px;
}}

.stSelectbox > div > div {{
    background-color: #111111;
    border: 1px solid {NEON_PINK};
    transition: all 0.3s;
}}

.stSelectbox > div > div:hover {{
    border: 1px solid {NEON_GREEN};
    box-shadow: 0 0 10px rgba(0, 255, 95, 0.5);
}}

/* Sidebar styling */
.css-1d391kg, .css-163ttbj, [data-testid="stSidebar"] {{
    background-color: #0A0A0A;
    border-right: 1px solid rgba(255, 0, 135, 0.2);
}}

/* Sidebar elements hover effects */
.css-1d391kg:hover, .css-163ttbj:hover {{
    background-color: #0F0F0F;
}}

/* Text input styling */
.stTextInput > div > div > input {{
    background-color: #111111;
    color: white;
    border: 1px solid {NEON_PINK};
    border-radius: 5px;
    transition: all 0.3s;
}}

.stTextInput > div > div > input:focus {{
    border: 1px solid {NEON_GREEN};
    box-shadow: 0 0 10px rgba(0, 255, 95, 0.5);
}}

/* File uploader styling */
.stFileUploader > div {{
    background-color: #111111;
    border: 2px dashed {NEON_PINK};
    border-radius: 5px;
    transition: all 0.3s;
}}

.stFileUploader > div:hover {{
    border-color: {NEON_GREEN};
    background-color: rgba(0, 255, 95, 0.05);
}}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    background-color: #111111;
    border-radius: 5px 5px 0 0;
    border: 1px solid rgba(255, 0, 135, 0.3);
    border-bottom: none;
    color: white;
    transition: all 0.3s;
}}

.stTabs [data-baseweb="tab"]:hover {{
    background-color: rgba(255, 0, 135, 0.2);
    transform: translateY(-2px);
}}

.stTabs [aria-selected="true"] {{
    background-color: rgba(255, 0, 135, 0.3);
    border-color: {NEON_PINK};
}}

/* Card effects for container elements */
div.element-container:hover, div.css-ocqkz7:hover, div.css-keje6w:hover, div.css-12oz5g7:hover {{
    /* Subtle lift effect removed to prevent jumpiness, focus on hover-card */
    /* transform: translateY(-2px); */
}}

/* Metric elements styling */
[data-testid="stMetricValue"] {{
    background-color: rgba(0, 0, 0, 0.6);
    border-radius: 5px;
    padding: 10px;
    border-left: 3px solid {NEON_PINK};
    transition: all 0.3s;
}}

[data-testid="stMetricValue"]:hover {{
    border-left: 3px solid {NEON_GREEN};
    background-color: rgba(0, 0, 0, 0.8);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}}

/* Pulsing animation */
@keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(255, 0, 135, 0.7); }}
    70% {{ box-shadow: 0 0 0 10px rgba(255, 0, 135, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(255, 0, 135, 0); }}
}}

.pulse {{
    animation: pulse 2s infinite;
}}

/* Custom scrollbar */
::-webkit-scrollbar {{
    width: 10px;
    height: 10px;
}}

::-webkit-scrollbar-track {{
    background: #111111;
}}

::-webkit-scrollbar-thumb {{
    background: {NEON_PINK};
    border-radius: 5px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {NEON_GREEN};
}}

/* Hover card effect */
.hover-card {{
    background-color: #111;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid rgba(255,0,135,0.3);
    transition: all 0.3s ease-in-out; /* Smoother transition */
}}

.hover-card:hover {{
    border-color: {NEON_GREEN};
    transform: translateY(-5px) scale(1.01); /* Add subtle scale */
    box-shadow: 0 10px 20px rgba(0,0,0,0.4); /* Darker shadow */
}}

/* Gradient border effect */
.gradient-border {{
    position: relative;
    padding: 15px;
    margin: 10px 0;
    border-radius: 10px;
    background: linear-gradient(to right, black, #111111);
}}

.gradient-border::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border-radius: 10px;
    padding: 2px;
    background: linear-gradient(to right, {NEON_PINK}, {NEON_GREEN});
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}}

/* Dashboard section styling */
.dashboard-section {{
    margin: 20px 0;
    padding: 15px;
    border-radius: 10px;
    background-color: rgba(10, 10, 10, 0.7); /* Darker base */
    border: 1px solid rgba(255, 255, 255, 0.1); /* Subtle border */
    transition: all 0.3s;
    backdrop-filter: blur(2px); /* Glassmorphism hint */
}}

.dashboard-section:hover {{
    background-color: rgba(20, 20, 20, 0.8);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}}

/* Title hover effect */
.main-title-container {{
    display: flex;
    align-items: center;
}}
.main-title {{
    color: {NEON_PINK};
    margin-bottom: 0;
    font-family: "Courier New", monospace;
    transition: all 0.3s ease-in-out;
    text-shadow: 0 0 5px {NEON_PINK}, 0 0 10px {NEON_PINK};
    cursor: default; /* Indicate it's not clickable */
    display: inline-block; /* Needed for transform */
}}

.main-title-container:hover .main-title {{
    color: {NEON_GREEN};
    text-shadow: 0 0 8px {NEON_GREEN}, 0 0 15px {NEON_GREEN}, 0 0 20px white;
    transform: scale(1.02);
}}

/* Animated Symbol Background */
@keyframes rotateShapes {{
    0% {{ transform: rotate(0deg) scale(1); opacity: 0.8; }}
    50% {{ transform: rotate(5deg) scale(1.03); opacity: 1; }}
    100% {{ transform: rotate(0deg) scale(1); opacity: 0.8; }}
}}

.animated-symbol-bg {{
    width: 65px; /* Slightly larger */
    height: 65px;
    position: relative;
    display: inline-block;
    vertical-align: middle;
    margin-right: 15px;
    animation: rotateShapes 12s ease-in-out infinite;
    transition: transform 0.4s ease;
    filter: drop-shadow(0 0 5px rgba(255, 0, 135, 0.5)); /* Add a base glow */
}}

.animated-symbol-bg:hover {{
    transform: scale(1.15) rotate(-8deg);
    filter: drop-shadow(0 0 10px rgba(0, 255, 95, 0.7)); /* Hover glow */
}}

.animated-symbol-bg svg {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}}

/* ML Pipeline Steps Styling */
.ml-step {{
    text-align:center;
    padding: 15px 10px; /* Increased padding */
    background-color: rgba(20, 20, 20, 0.7); /* Darker, more contrast */
    border: 1px solid rgba(255, 0, 135, 0.4); /* Slightly stronger border */
    border-radius: 8px;
    height: 160px; /* Further increased height */
    display: flex;
    flex-direction: column;
    justify-content: space-between; /* Space out content */
    transition: all 0.3s ease-in-out;
    word-wrap: break-word; /* Ensure text wraps */
    overflow: hidden; /* Hide overflow */
    backdrop-filter: blur(2px);
}}
.ml-step:hover {{
    border-color: {NEON_GREEN};
    transform: translateY(-5px) scale(1.02); /* Lift and scale */
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
}}
.ml-step p {{
    margin: 5px 0;
    font-size: 0.9em;
    line-height: 1.3;
    color: #e0e0e0; /* Slightly brighter text */
}}
.ml-step .step-title {{
    font-weight: bold;
    color: white;
    margin-bottom: 10px; /* More space below title */
}}
.ml-step .status-icon {{
    font-size: 32px; /* Larger icon */
    margin-top: auto;
    padding-bottom: 5px;
}}
.ml-step .status-message {{
    font-size: 0.8em;
    color: #a0a0a0; /* Dimmer status text */
    margin-top: 5px;
    min-height: 2.4em; /* Reserve space for 2 lines */
    line-height: 1.2;
}}


@keyframes pulse-success {{
    0% {{ box-shadow: 0 0 0 0 rgba(0, 255, 95, 0.7); }}
    70% {{ box-shadow: 0 0 0 10px rgba(0, 255, 95, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(0, 255, 95, 0); }}
}}
.ml-step.success {{
    border-color: {NEON_GREEN};
    background-color: rgba(0, 255, 95, 0.15); /* Slightly more visible success bg */
    animation: pulse-success 1.5s ease-out;
}}
.ml-step.success .status-message {{
     color: {NEON_GREEN}; /* Green status text on success */
}}

/* Feature Importance Bar Styling */
.feature-bar {{
    height: 20px;
    background: linear-gradient(to right, {NEON_PINK}aa, {NEON_GREEN}aa); /* Added alpha */
    border-radius: 5px;
    margin-bottom: 5px;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1); /* Subtle border */
}}
.feature-bar::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(90deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 50%, rgba(255,255,255,0.1) 100%);
    opacity: 0.6;
}}
.feature-bar:hover {{
    transform: scaleX(1.01);
    box-shadow: 0 0 8px rgba(255, 0, 135, 0.4);
    background: linear-gradient(to right, {NEON_PINK}, {NEON_GREEN}); /* Brighter on hover */
}}
.feature-container {{
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    padding: 5px 0;
}}
.feature-name {{
    width: 150px; /* Increased width */
    text-align: right;
    padding-right: 15px;
    font-size: 13px;
    color: #ccc;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.feature-value {{
    margin-left: 10px;
    font-weight: bold;
    font-size: 13px;
    color: white;
    min-width: 40px; /* Ensure space for value */
}}
.feature-bar-wrapper {{
    flex-grow: 1;
}}

/* Styling for the quote display in sidebar */
.quote-box {{
    background-color: rgba(30, 30, 30, 0.8); /* Slightly transparent */
    border-left: 4px solid {NEON_PINK};
    padding: 12px 18px; /* More padding */
    margin-top: 15px;
    border-radius: 5px;
    font-style: italic;
    color: #ccc;
    font-size: 0.9em;
    transition: all 0.3s ease-in-out;
    backdrop-filter: blur(2px);
}}
.quote-box:hover {{
    border-left-color: {NEON_GREEN};
    color: white;
}}

/* Metrics Card Layout */
.metrics-container {{
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 20px;
}}
.metric-card {{
    flex: 1;
    min-width: 180px;
    background-color: rgba(17, 17, 17, 0.8); /* Slightly transparent */
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    transition: all 0.3s;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-left: 4px solid transparent; /* Keep border for consistency */
    backdrop-filter: blur(2px);
}}
.metric-card:hover {{
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    background-color: rgba(25, 25, 25, 0.9);
}}
.metric-card.up {{ border-left-color: {NEON_GREEN}; }}
.metric-card.down {{ border-left-color: red; }}
.metric-card.neutral {{ border-left-color: {NEON_PINK}; }}

.metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
.metric-title {{ color: #999; font-size: 14px; margin-bottom: 5px; text-transform: uppercase; }}
.metric-subtext {{ color: #777; font-size: 0.85em; }}

/* Styling for Prediction Verdict */
.verdict-box {{
    background-color: rgba(17, 17, 17, 0.9);
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
    border-left: 5px solid {NEON_PINK};
    transition: all 0.3s ease-in-out;
    backdrop-filter: blur(3px);
}}
.verdict-box:hover {{
    /* border-left-color: {NEON_GREEN}; */ /* Keep original verdict color on hover */
    box-shadow: 0 8px 15px rgba(0,0,0,0.4);
}}
.verdict-title {{
    color: white;
    font-weight: bold;
    font-size: 1.3em;
    margin-bottom: 10px;
}}
.verdict-text {{
    color: #ccc;
    font-size: 1.0em;
    line-height: 1.5;
}}
.verdict-icon {{
    font-size: 2em;
    margin-right: 10px;
    vertical-align: middle;
}}

</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

# Function to create animated symbol background SVG for header
def create_animated_symbol_svg():
    svg = f"""
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <filter id="glow">
                <feGaussianBlur stdDeviation="1.5" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
        <g style="filter: url(#glow); opacity: 0.9;">
            <!-- Circle -->
            <circle cx="30" cy="30" r="12" fill="none" stroke="{NEON_PINK}" stroke-width="2">
                <animate attributeName="r" values="12;14;12" dur="4s" repeatCount="indefinite" />
                <animate attributeName="stroke" values="{NEON_PINK};{NEON_GREEN};{NEON_PINK}" dur="6s" repeatCount="indefinite" />
            </circle>
            <!-- Triangle -->
            <polygon points="70,20 85,45 55,45" fill="none" stroke="{NEON_GREEN}" stroke-width="2">
                 <animateTransform attributeName="transform" type="rotate" from="0 70 32.5" to="360 70 32.5" dur="15s" repeatCount="indefinite"/>
                 <animate attributeName="stroke" values="{NEON_GREEN};{NEON_PINK};{NEON_GREEN}" dur="7s" repeatCount="indefinite" />
            </polygon>
            <!-- Square -->
            <rect x="20" y="60" width="25" height="25" fill="none" stroke="{NEON_PINK}" stroke-width="2">
                 <animate attributeName="stroke" values="{NEON_PINK};{NEON_GREEN};{NEON_PINK}" dur="5s" repeatCount="indefinite" />
                 <animate attributeName="x" values="20;18;20" dur="8s" repeatCount="indefinite" />
                 <animate attributeName="y" values="60;62;60" dur="8s" repeatCount="indefinite" />
            </rect>
            <!-- Umbrella (Simplified Path) -->
            <path d="M 65 88 L 65 75 Q 77.5 60 90 75 L 90 88 Z" fill="none" stroke="{NEON_GREEN}" stroke-width="2" transform="translate(-5, -5)">
                 <animate attributeName="stroke" values="{NEON_GREEN};{NEON_PINK};{NEON_GREEN}" dur="8s" repeatCount="indefinite" />
                 <animateTransform attributeName="transform" type="translate" values="-5,-5; -7,-7; -5,-5" dur="9s" repeatCount="indefinite"/>
            </path>
        </g>
    </svg>
    """
    return svg

# Custom header with improved title and symbol
def squid_header():
    st.markdown(f"""
    <div class='main-title-container'>
        <div class='animated-symbol-bg'>{create_animated_symbol_svg()}</div>
        <div>
            <h1 class='main-title'>SQUIDSTOCK SURVIVAL</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 5px; border-top: 1px solid rgba(255, 0, 135, 0.3);'>", unsafe_allow_html=True)


# Function to fetch stock data
@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def load_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        if data.empty:
             return None, None, f"No data found for ticker '{ticker}'. It might be delisted, invalid, or lack data for the selected period."

        # Basic Data Cleaning: Remove rows where Close is NaN before filling
        data.dropna(subset=['Close'], inplace=True)

        # Check again if empty after dropping NaNs
        if data.empty:
            return None, None, f"No valid 'Close' price data found for ticker '{ticker}' in the selected period."


        # Forward fill first, then backfill for robustness
        # Fill OHLC with previous close if missing, Volume with 0
        data['Open'] = data['Open'].fillna(data['Close'].ffill())
        data['High'] = data['High'].fillna(data['Close'].ffill())
        data['Low'] = data['Low'].fillna(data['Close'].ffill())
        data['Volume'] = data['Volume'].fillna(0)

        # Forward fill then backfill remaining (mostly for early data points)
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        info = stock.info

        # Final check for NaNs after filling
        if data.isnull().values.any():
            nan_cols = data.columns[data.isnull().any()].tolist()
            st.warning(f"Data for {ticker} still contains missing values after fill in columns: {nan_cols}. Check source data. Proceeding might yield unexpected results.")
            # Decide how to handle remaining NaNs: drop rows, columns, or impute differently
            # For now, we proceed but warn the user.
            # data.dropna(inplace=True) # Option: drop rows with any remaining NaNs

        # Ensure data types are correct
        for col in ['Open', 'High', 'Low', 'Close']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0).astype(np.int64)
        data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # Drop if conversion failed

        if data.empty:
             return None, None, f"Data for ticker '{ticker}' became empty after cleaning/type conversion."


        return data, info, None # Return None for error if successful

    except Exception as e:
        error_msg = f"Error fetching data for '{ticker}': {str(e)}"
        if "Too many requests" in str(e):
            error_msg = "API Limit Reached: Too many requests to Yahoo Finance. Please wait a while before trying again or use a CSV file."
        elif "No data found" in str(e) or "symbol may be delisted" in str(e) or "failed" in str(e).lower():
             error_msg = f"Invalid Ticker or No Data: Could not find data for '{ticker}'. Please check the symbol and selected period."

        # Don't show error here, return it to be handled in main()
        # st.error(error_msg)
        return None, None, error_msg # Return the error message


# Function to prepare data for ML models
def prepare_ml_data(df):
    min_data_length = 60 # Need at least 50 for longest MA + 5 target shift + buffer
    if df is None or len(df) < min_data_length:
        return None, None, None, None, None, None, None, f"Insufficient data: Need at least {min_data_length} trading days for robust feature calculation and prediction. Found {len(df) if df is not None else 0}. Try a longer period."

    df_processed = df.copy()
    df_processed.sort_index(inplace=True) # Ensure chronological order

    # Feature Engineering
    windows = [5, 10, 20, 50]
    features_created = []
    for w in windows:
        df_processed[f'MA{w}'] = df_processed['Close'].rolling(window=w, min_periods=max(1, w//2)).mean() # Use min_periods
        df_processed[f'StdDev{w}'] = df_processed['Close'].rolling(window=w, min_periods=max(1, w//2)).std()
        df_processed[f'Momentum{w}'] = df_processed['Close'].diff(w) # Use diff for momentum
        features_created.extend([f'MA{w}', f'StdDev{w}', f'Momentum{w}'])

    df_processed['Volatility10'] = df_processed['Close'].pct_change().rolling(window=10, min_periods=5).std() * np.sqrt(252) # Annualized volatility
    df_processed['RSI14'] = calculate_rsi(df_processed['Close'], 14)
    features_created.extend(['Volatility10', 'RSI14'])

    # Handle initial NaNs created by rolling functions more robustly
    df_processed.bfill(inplace=True) # Backfill first to propagate values backward
    df_processed.ffill(inplace=True) # Then forward fill

    # Drop any remaining NaNs (should be minimal if bfill/ffill worked)
    initial_rows = len(df_processed)
    df_processed.dropna(inplace=True)
    if len(df_processed) < initial_rows:
         st.info(f"Dropped {initial_rows - len(df_processed)} rows with persistent NaN values after feature engineering.")

    if len(df_processed) < 20: # Check again after cleaning
        return None, None, None, None, None, None, None, f"Not enough data remaining ({len(df_processed)} rows) after feature creation and cleaning."

    # Target variables (shifted)
    future_shift = 5
    df_processed['Target_Reg'] = df_processed['Close'].shift(-future_shift)
    df_processed['Target_Cls'] = (df_processed['Target_Reg'] > df_processed['Close']).astype(int)

    # Store features *before* dropping target NaNs for correlation heatmap later
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    all_feature_cols = base_features + features_created
    # Make sure all columns exist
    all_feature_cols = [col for col in all_feature_cols if col in df_processed.columns]
    full_feature_df = df_processed[all_feature_cols + ['Target_Reg', 'Target_Cls']].copy()

    # Drop rows where targets are NaN (the last 'future_shift' rows)
    df_processed = df_processed.dropna(subset=['Target_Reg', 'Target_Cls'])

    if df_processed.empty:
        return None, None, None, None, None, None, None, "No data left after creating target variables and cleaning."

    # Define features for modeling (exclude targets)
    X = df_processed[all_feature_cols]
    y_reg = df_processed['Target_Reg']
    y_cls = df_processed['Target_Cls']

    # Store full df *after* target creation and NaN drop for correlation
    # Re-calculate correlation df based on the final modeling data
    final_corr_df = X.copy()
    final_corr_df['Price_5d_Ahead'] = y_reg
    final_corr_df['Trend_5d_Ahead'] = y_cls

    # Split data (using time series split is better, but train_test_split is simpler here)
    # Use shuffle=False for time series data
    try:
        # Ensure test size isn't too large for short datasets
        test_fraction = 0.2
        if len(X) * test_fraction < 5: # Ensure at least a few test samples
            test_fraction = max(0.1, 5 / len(X)) if len(X) > 0 else 0.2

        X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=test_fraction, random_state=42, shuffle=False)
        # Align classification split with regression split indices
        y_cls_train = y_cls.loc[y_reg_train.index]
        y_cls_test = y_cls.loc[y_reg_test.index]

        # Check if splits are non-empty
        if X_train.empty or X_test.empty:
             return None, None, None, None, None, None, None, f"Data splitting resulted in empty train/test set (Total samples: {len(X)})."

    except Exception as e:
        return None, None, None, None, None, None, None, f"Data splitting error: {str(e)}. Indices might mismatch or data too short."

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Latest data for prediction
    latest_features = X.iloc[-1].values.reshape(1, -1)
    latest_data_scaled = scaler.transform(latest_features)

    # Return the dataframe for correlation analysis
    return X_train_scaled, X_test_scaled, y_reg_train, y_reg_test, y_cls_train, y_cls_test, latest_data_scaled, final_corr_df


# Calculate RSI (Relative Strength Index)
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    # Use Exponential Moving Average (EMA) for RSI calculation - more standard
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    # Handle division by zero if avg_loss is 0
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))


    # Convert to pandas Series to use fillna
    rsi_series = pd.Series(rsi, index=prices.index)

    # Handle potential NaNs at the beginning
    rsi_series = rsi_series.fillna(method='bfill').fillna(50) # Backfill then fill with neutral 50
    return rsi_series


# Train ML models and make predictions
def train_and_predict(X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, latest_data):
    results = {}
    models = {} # Store trained models

    if X_train is None or X_test is None or y_reg_train is None or y_cls_train is None or latest_data is None:
        return None, None, "Input data for training/prediction is missing."
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
         return None, None, "Train or test set is empty after preparation."

    try:
        # --- Linear Regression ---
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_reg_train)
        models['regression'] = reg_model

        y_reg_pred_test = reg_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_test))
        r2 = r2_score(y_reg_test, y_reg_pred_test)

        next_price_pred = reg_model.predict(latest_data)[0]
        # Calculate prediction std based on test set residuals
        residuals_test = y_reg_test.values - y_reg_pred_test
        if len(residuals_test) > 1:
             prediction_std = np.std(residuals_test)
        else:
             prediction_std = 0 # Cannot calculate std dev from one residual

        lower_bound = next_price_pred - 1.96 * prediction_std
        upper_bound = next_price_pred + 1.96 * prediction_std

        results['regression'] = {
            'prediction': next_price_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'rmse': rmse,
            'r2': r2,
            'y_true': y_reg_test.values, # For plotting
            'y_pred': y_reg_pred_test, # For plotting
            'residuals': residuals_test # For plotting
        }

        # --- Logistic Regression ---
        # Check if there's more than one class in the training target
        if len(np.unique(y_cls_train)) > 1:
            # Handle potential imbalance with class_weight='balanced'
            cls_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000, solver='liblinear') # Use liblinear for small datasets
            cls_model.fit(X_train, y_cls_train)
            models['classification'] = cls_model

            y_cls_pred_test = cls_model.predict(X_test)
            y_cls_proba_test = cls_model.predict_proba(X_test)[:, 1] # Prob of class 1 (Up)
            accuracy = accuracy_score(y_cls_test, y_cls_pred_test)
            # Ensure confusion matrix works even if only one class predicted in test set
            conf_matrix = confusion_matrix(y_cls_test, y_cls_pred_test, labels=[0, 1])

            trend_prob = cls_model.predict_proba(latest_data)[0][1]
            trend_pred_class = 1 if trend_prob >= 0.5 else 0
            trend_prediction = "UP" if trend_pred_class == 1 else "DOWN"

            results['classification'] = {
                'trend_numeric': trend_pred_class,
                'trend': trend_prediction,
                'probability': trend_prob,
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'y_true': y_cls_test.values, # For plotting
                'y_pred': y_cls_pred_test, # For plotting
                'y_proba': y_cls_proba_test # For ROC curve etc. (optional)
            }
        else:
            # Handle case where only one class exists in training data
            majority_class = y_cls_train.iloc[0]
            trend_prediction = "UP" if majority_class == 1 else "DOWN"
            st.warning(f"Logistic Regression Warning: Only one class ({trend_prediction}) present in the training data. Classification model cannot be reliably trained. Predicting majority class.")
            results['classification'] = {
                'trend_numeric': majority_class,
                'trend': trend_prediction,
                'probability': 1.0 if majority_class == 1 else 0.0,
                'accuracy': None, # Cannot calculate meaningful accuracy
                'confusion_matrix': None,
                'y_true': y_cls_test.values,
                'y_pred': np.full(len(y_cls_test), majority_class), # Predict majority class for all test samples
                'y_proba': np.full(len(y_cls_test), 1.0 if majority_class == 1 else 0.0)
            }
            models['classification'] = None # No trained model

        return results, models, None # Return results, models, and no error

    except Exception as e:
        st.error(f"Error during ML model training/prediction: {e}")
        st.error(traceback.format_exc()) # Print full traceback for debugging
        return None, None, f"Error during model training or prediction: {str(e)}"


# Function to display stock charts with enhanced interactivity
def display_stock_charts(data, ticker):
    df_chart = data.copy()
    df_chart['MA20'] = df_chart['Close'].rolling(window=20).mean()
    df_chart['MA50'] = df_chart['Close'].rolling(window=50).mean()
    df_chart['RSI14'] = calculate_rsi(df_chart['Close'], 14) # Add RSI

    # --- Price Chart ---
    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Candlestick trace
    fig_price.add_trace(go.Candlestick(
        x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
        low=df_chart['Low'], close=df_chart['Close'], name='Price',
        increasing_line_color= NEON_GREEN, decreasing_line_color= 'red'
    ), row=1, col=1)

    # Moving Averages trace
    fig_price.add_trace(go.Scatter(
        x=df_chart.index, y=df_chart['MA20'], name='MA 20',
        line=dict(color=NEON_GREEN, width=1.5)
    ), row=1, col=1)
    fig_price.add_trace(go.Scatter(
        x=df_chart.index, y=df_chart['MA50'], name='MA 50',
        line=dict(color=NEON_PINK, width=1.5)
    ), row=1, col=1)

    # RSI trace
    fig_price.add_trace(go.Scatter(
        x=df_chart.index, y=df_chart['RSI14'], name='RSI 14',
        line=dict(color='cyan', width=1)
    ), row=2, col=1)
    # Add RSI overbought/oversold lines
    fig_price.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig_price.add_hline(y=30, line_dash="dash", line_color=NEON_GREEN, opacity=0.5, row=2, col=1)

    # Update layout for Price + RSI Chart
    fig_price.update_layout(
        title={'text': f"{ticker} Stock Price, MAs & RSI", 'font': {'size': 22, 'color': 'white'}, 'x': 0.5, 'xanchor': 'center'},
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600, # Increased height for two subplots
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0.5)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,17,17,0.9)',
        yaxis1_title="Price (USD)", # Title for first y-axis (Price)
        yaxis2_title="RSI", # Title for second y-axis (RSI)
        xaxis_showticklabels=True, xaxis2_showticklabels=True # Ensure x-axis labels show
    )
    fig_price.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig_price.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)


    # --- Volume Chart (Separate) ---
    volume_colors = np.where(df_chart['Close'] >= df_chart['Open'], NEON_GREEN, 'red')
    fig_volume = go.Figure(data=[go.Bar(
        x=df_chart.index, y=df_chart['Volume'], marker_color=volume_colors, name='Volume'
    )])

    df_chart['VolumeMA20'] = df_chart['Volume'].rolling(window=20).mean()
    fig_volume.add_trace(go.Scatter(
        x=df_chart.index, y=df_chart['VolumeMA20'], name='Volume MA 20',
        line=dict(color=NEON_PINK, width=1.5, dash='dash')
    ))

    fig_volume.update_layout(
        title={'text': f"{ticker} Trading Volume", 'font': {'size': 20, 'color': 'white'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title="Volume", template="plotly_dark", height=300, hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(0,0,0,0.5)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,17,17,0.9)'
    )

    return fig_price, fig_volume

# Function to display price prediction
def display_price_prediction(last_price, prediction_data, ticker):
    # Safely get prediction values
    pred_reg = prediction_data.get('regression', {})
    next_price = pred_reg.get('prediction', last_price) # Default to last price if missing
    lower_bound = pred_reg.get('lower_bound', last_price * 0.95) # Default +/- 5%
    upper_bound = pred_reg.get('upper_bound', last_price * 1.05)
    rmse = pred_reg.get('rmse')
    r2 = pred_reg.get('r2')

    # Use .item() if numpy float
    if isinstance(next_price, np.number): next_price = next_price.item()
    if isinstance(lower_bound, np.number): lower_bound = lower_bound.item()
    if isinstance(upper_bound, np.number): upper_bound = upper_bound.item()
    if isinstance(rmse, np.number): rmse = rmse.item()
    if isinstance(r2, np.number): r2 = r2.item()


    pct_change = ((next_price - last_price) / last_price) * 100 if last_price else 0

    # Create prediction visualization
    last_date = datetime.now()
    future_dates = [last_date + timedelta(days=i) for i in range(6)] # Assuming 5 days ahead prediction target

    fig = go.Figure()

    # Last actual price
    fig.add_trace(go.Scatter(
        x=[future_dates[0]], y=[last_price], mode='markers',
        marker=dict(color='white', size=12, symbol='diamond-open', line=dict(width=1, color='white')),
        name='Last Actual Price'
    ))

    # Predicted price point at day 5
    fig.add_trace(go.Scatter(
        x=[future_dates[5]], y=[next_price], mode='markers',
        marker=dict(color=NEON_PINK, size=12, symbol='star', line=dict(width=1, color='white')),
        name=f'Predicted Price ({future_dates[5]:%Y-%m-%d})'
    ))

    # Line connecting last price to prediction
    fig.add_trace(go.Scatter(
        x=[future_dates[0], future_dates[5]], y=[last_price, next_price],
        mode='lines', line=dict(color=NEON_PINK, width=2, dash='dash'),
        showlegend=False
    ))

    # Prediction interval (vertical line or shaded area at day 5)
    fig.add_trace(go.Scatter(
        x=[future_dates[5], future_dates[5]], y=[lower_bound, upper_bound],
        mode='lines', line=dict(color=NEON_GREEN, width=5, dash='solid'),
        name='95% Confidence Interval', opacity=0.7
    ))
    # Add markers for bounds
    fig.add_trace(go.Scatter(x=[future_dates[5]], y=[upper_bound], mode='markers', marker=dict(symbol='line-ew-open', color=NEON_GREEN, size=15), showlegend=False))
    fig.add_trace(go.Scatter(x=[future_dates[5]], y=[lower_bound], mode='markers', marker=dict(symbol='line-ew-open', color=NEON_GREEN, size=15), showlegend=False))


    fig.update_layout(
        title=f"5-Day Price Prediction for {ticker} (Linear Regression)",
        xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark", height=400,
        xaxis=dict(range=[future_dates[0] - timedelta(days=1), future_dates[5] + timedelta(days=1)]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(0,0,0,0.5)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,17,17,0.9)'
    )

    # --- Display prediction stats ---
    col1, col2, col3 = st.columns(3)

    pct_color = NEON_GREEN if pct_change > 0 else "red" if pct_change < 0 else "white"
    direction = "📈" if pct_change > 0 else "📉" if pct_change < 0 else "➖"

    with col1:
        st.markdown(f"""
        <div class='metric-card {'up' if pct_change > 0 else 'down' if pct_change < 0 else 'neutral'}' style='border-left-color:{pct_color};'>
            <div class='metric-title'>Predicted Price</div>
            <div class='metric-value' style='color:{pct_color};'>{direction} ${next_price:.2f}</div>
            <div class='metric-subtext' style='color:{pct_color};'>({pct_change:+.2f}% vs last)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card neutral'>
            <div class='metric-title'>Prediction Range</div>
            <div class='metric-value' style='font-size: 18px; color:white;'>${lower_bound:.2f} - ${upper_bound:.2f}</div>
            <div class='metric-subtext'>95% Confidence Interval</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        rmse_text = f"${rmse:.3f}" if rmse is not None and not np.isnan(rmse) else "N/A" # More precision for RMSE
        r2_text = f"{r2:.3f}" if r2 is not None and not np.isnan(r2) else "N/A"
        st.markdown(f"""
        <div class='metric-card neutral'>
            <div class='metric-title'>Model Fit (Test Set)</div>
            <div class='metric-value' style='font-size: 18px; color:white;'>RMSE: {rmse_text}</div>
            <div class='metric-subtext'>R² Score: {r2_text}</div>
        </div>
        """, unsafe_allow_html=True)

    return fig

# Function to display trend prediction
def display_trend_prediction(prediction_data):
    # Safely get classification results
    pred_cls = prediction_data.get('classification', {})
    trend = pred_cls.get('trend', 'N/A')
    probability = pred_cls.get('probability', 0.5) # Default to 0.5 if missing
    accuracy = pred_cls.get('accuracy')
    conf_matrix = pred_cls.get('confusion_matrix')

    # Use .item() if numpy float
    if isinstance(probability, np.number): probability = probability.item()
    if isinstance(accuracy, np.number): accuracy = accuracy.item()


    prob_pct = probability * 100
    accuracy_pct = accuracy * 100 if accuracy is not None and not np.isnan(accuracy) else None

    color = NEON_GREEN if trend == "UP" else "red" if trend == "DOWN" else "gray"
    icon = "⬆️" if trend == "UP" else "⬇️" if trend == "DOWN" else "❔"

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class='hover-card' style='text-align:center; border-color:{color}; height: 220px; display: flex; flex-direction: column; justify-content: center;'>
            <h4 style='color:{NEON_PINK}; margin-bottom:5px;'>Predicted Trend</h4>
            <p style='font-size:60px; margin:5px 0; color:{color};'>{icon}</p>
            <p style='font-size:22px; margin-top:5px; color:{color}; font-weight:bold;'>{trend}</p>
            <p style='margin:0; color:gray; font-size:0.9em;'>5-Day Forecast</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Gauge chart for probability
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob_pct,
            title={'text': "Prediction Confidence", 'font': {'color': 'white', 'size': 14}},
            gauge={'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'size':10}},
                   'bar': {'color': color, 'thickness': 0.7},
                   'bgcolor': 'rgba(50, 50, 50, 0.3)',
                   'borderwidth': 1, 'bordercolor': "#444",
                   'steps': [{'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.2)'},
                             {'range': [50, 100], 'color': 'rgba(0, 255, 95, 0.2)'}],
                   'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.9, 'value': 50}},
            number={'suffix': "%", 'font': {'color': 'white', 'size': 24}}
        ))
        fig_gauge.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20),
                                paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- Display model accuracy and Confusion Matrix ---
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True) # Add spacing
    with st.expander("Trend Model Performance Details (Test Set)", expanded=False):
        col_acc, col_cm = st.columns(2)
        with col_acc:
            acc_text = f"{accuracy_pct:.1f}%" if accuracy_pct is not None else "N/A"
            st.metric(label="Model Accuracy", value=acc_text)
            st.caption("How often the model predicted the correct trend (Up/Down) on unseen historical data.")

        with col_cm:
            if conf_matrix is not None and isinstance(conf_matrix, np.ndarray) and conf_matrix.shape == (2, 2):
                 # Use a standard Plotly colorscale for confusion matrix
                 fig_cm = px.imshow(conf_matrix, text_auto=True, aspect="auto",
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['Down', 'Up'], y=['Down', 'Up'],
                                   color_continuous_scale='rdbu') # Use a built-in colorscale
                 fig_cm.update_layout(title="Confusion Matrix", template="plotly_dark",
                                      height=250, margin=dict(l=10, r=10, t=40, b=10),
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,17,17,0.9)',
                                      coloraxis_showscale=False)
                 fig_cm.update_xaxes(side="top", title_standoff=10)
                 fig_cm.update_yaxes(title_standoff=10)
                 st.plotly_chart(fig_cm, use_container_width=True)
            elif pred_cls.get('accuracy') is None and 'classification' in prediction_data: # Check if it was the single-class case
                 st.info("Confusion matrix not applicable as only one class was present in training data.")
            else:
                 st.info("Confusion matrix data not available or invalid.")


# Function to display company info
def display_company_info(info):
    if not info or not isinstance(info, dict):
        st.sidebar.warning("Company information could not be retrieved.")
        return

    # Extract data safely using .get()
    company_name = info.get('longName', 'N/A')
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    market_cap = info.get('marketCap')
    pe_ratio = info.get('trailingPE')
    website = info.get('website', '#')
    summary = info.get('longBusinessSummary', 'No business summary available.')

    # Format market cap
    market_cap_str = "N/A"
    if isinstance(market_cap, (int, float)):
        if market_cap >= 1e12: market_cap_str = f"${market_cap/1e12:.2f} Trillion"
        elif market_cap >= 1e9: market_cap_str = f"${market_cap/1e9:.2f} Billion"
        elif market_cap >= 1e6: market_cap_str = f"${market_cap/1e6:.2f} Million"
        else: market_cap_str = f"${market_cap:,.0f}"

    # Format P/E Ratio
    pe_ratio_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"

    # Format website link
    website_link = f"http://{website}" if website and website != '#' and not website.startswith(('http://', 'https://')) else website if website != '#' else '#'
    website_display = website if website and website != '#' else 'N/A'

    # Display in sidebar
    st.sidebar.markdown(f"""
    <div class='hover-card' style='background-color:#0A0A0A; border-color: rgba(255,255,255,0.1);'>
        <h4 style='color:{NEON_PINK}; margin-bottom:10px;'>{company_name}</h4>
        <p style='margin:2px 0; font-size:0.9em;'><strong style='color:{NEON_GREEN};'>Sector:</strong> {sector}</p>
        <p style='margin:2px 0; font-size:0.9em;'><strong style='color:{NEON_GREEN};'>Industry:</strong> {industry}</p>
        <p style='margin:2px 0; font-size:0.9em;'><strong style='color:{NEON_GREEN};'>Market Cap:</strong> {market_cap_str}</p>
        <p style='margin:2px 0; font-size:0.9em;'><strong style='color:{NEON_GREEN};'>P/E Ratio:</strong> {pe_ratio_str}</p>
        <p style='margin:2px 0; font-size:0.9em;'><strong style='color:{NEON_GREEN};'>Website:</strong> <a href='{website_link}' target='_blank' style='color:white; text-decoration:underline;'>{website_display}</a></p>
        <details style='margin-top:10px;'>
            <summary style='color:{NEON_PINK}; cursor:pointer; font-size:0.9em;'>Business Summary</summary>
            <p style='font-size:0.85em; color:#ccc; margin-top:5px; max-height: 150px; overflow-y: auto;'>{summary}</p>
        </details>
    </div>
    """, unsafe_allow_html=True)

# Function to display loading message (more subtle)
def loading_message(message="Processing..."):
    # Use st.spinner for better integration
    # This function might not be needed if st.spinner is used directly
    pass # Replaced by st.spinner context manager

# Function to create a base64 encoded image for welcome screen background
def get_base64_squid_game_bg():
    try:
        img = Image.new('RGB', (600, 300), color=DARK_BG)
        draw = ImageDraw.Draw(img)
        alpha = 70 # More subtle shapes

        # Grid lines
        for i in range(0, 600, 40): draw.line([(i, 0), (i, 300)], fill=(20, 20, 20), width=1)
        for i in range(0, 300, 40): draw.line([(0, i), (600, i)], fill=(20, 20, 20), width=1)

        # Shapes (more transparent) - Correct alpha application for Pillow
        def hex_to_rgba(hex_color, alpha_int):
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return rgb + (alpha_int,)

        draw.ellipse((40, 40, 120, 120), outline=hex_to_rgba(NEON_PINK, alpha), width=3)
        draw.polygon([(280, 40), (340, 120), (220, 120)], outline=hex_to_rgba(NEON_GREEN, alpha), width=3)
        draw.rectangle((450, 40, 550, 140), outline=hex_to_rgba(NEON_PINK, alpha), width=3)
        # Pillow arc needs bounding box [x0, y0, x1, y1]
        draw.arc((250, 180, 350, 280), start=0, end=180, fill=hex_to_rgba(NEON_GREEN, alpha), width=3) # Use fill for arc line in Pillow >= 8
        draw.line([(300, 230), (300, 280)], fill=hex_to_rgba(NEON_GREEN, alpha), width=3)


        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"Error generating background image: {e}")
        return ""

# Function to show welcome message
def show_welcome():
    img_str = get_base64_squid_game_bg()
    bg_style = f"""
        background-image: url(data:image/png;base64,{img_str});
        background-size: 200px 100px; /* Smaller tile size */
        background-repeat: repeat;
        padding: 40px; border-radius: 15px;
        border: 1px solid rgba(255,0,135,0.2);
        background-color: rgba(0,0,0,0.85); /* Darker overlay */
        text-align: center; margin-bottom: 30px;
    """
    st.markdown(f"""
    <style>
    /* Styles specific to welcome screen */
    @keyframes glow {{ 0% {{ text-shadow: 0 0 8px {NEON_PINK}; }} 50% {{ text-shadow: 0 0 18px {NEON_PINK}, 0 0 25px {NEON_PINK}; }} 100% {{ text-shadow: 0 0 8px {NEON_PINK}; }} }}
    .welcome-title {{ animation: glow 2.5s ease-in-out infinite; font-family: 'Courier New', monospace; color:{NEON_PINK}; font-size:3em; margin-bottom: 10px; }}
    .welcome-card {{ border-radius: 10px; padding: 20px; margin: 15px 0; background-color: rgba(17,17,17,0.8); border: 1px solid rgba(255,0,135,0.3); transition: all 0.3s; backdrop-filter: blur(3px); }}
    .welcome-card:hover {{ transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.4); border-color: rgba(255,0,135,0.8); }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='{bg_style}'>", unsafe_allow_html=True)
    st.markdown("<h1 class='welcome-title'>Welcome to SquidStock Survival</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:white; font-size:1.2em;'>Enter a stock ticker or upload a CSV in the sidebar to begin.</p>", unsafe_allow_html=True)
    st.markdown(f"<div style='margin:30px 0; display:inline-block;'>{create_animated_symbol_svg()}</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:white; font-size:1.1em;'>Will your analysis survive the market's game?</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center; color:white; margin-top:20px;'>Key Features</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='welcome-card'><h3 style='color:{NEON_PINK};'>🎯 Stock Analysis</h3><p>View raw data, interactive charts (Candlesticks, MAs, RSI, Volume), and key stats.</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='welcome-card'><h3 style='color:{NEON_GREEN};'>📊 ML Predictions</h3><p>5-day Price (Linear Regression) & Trend (Logistic Regression) forecasts with combined verdict.</p></div>", unsafe_allow_html=True) # Added verdict mention
    with col2:
        st.markdown(f"<div class='welcome-card'><h3 style='color:{NEON_PINK};'>📁 Data Flexibility</h3><p>Use Yahoo Finance API or upload your own historical stock data (CSV).</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='welcome-card'><h3 style='color:{NEON_GREEN};'>🔬 ML Insights</h3><p>Visualize the ML pipeline, explore feature influence, customize plot colors, and download predictions.</p></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='dashboard-section gradient-border' style='margin-top:30px; text-align:center;'>
        <h3 style='color:{NEON_PINK};'>How to Play</h3>
        <ol style='display:inline-block; text-align:left; color:white; padding-left: 20px; font-size: 0.95em;'>
            <li>Use the sidebar to choose data source (API or CSV).</li>
            <li>Enter a Ticker (e.g., AAPL, TSLA) or Upload your CSV file.</li>
            <li>Select the desired historical Time Period (for API).</li>
            <li>Click "Load / Refresh" or wait for CSV processing.</li>
            <li>Click a symbol (⭕ △ □ ☂) in the sidebar for a 'wisdom' quote!</li>
            <li>Explore 'Stock Analysis': View raw data, charts, and key stats.</li>
            <li>Go to 'ML Predictions': View forecasts, performance, and insights.</li>
            <li>Inside 'ML Predictions', select Price or Trend model focus.</li>
            <li>Customize plot colors using the options provided.</li>
            <li>Analyze the 'Prediction Verdict' for a combined outlook.</li>
            <li>Download the prediction summary using the download button.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin-top:40px; padding:20px; background-color:rgba(0,0,0,0.7); border-radius:10px; border-left:5px solid {NEON_PINK};'>
        <h3 style='color:{NEON_PINK};'>⚠️ DISCLAIMER</h3>
        <p style='color:white; font-size:0.9em;'>This application is purely for educational and entertainment purposes. Stock market predictions involve inherent uncertainty. Information and forecasts provided are based on historical data and statistical models, which may not accurately reflect future market conditions. Not financial advice. Always conduct your own research and consult with a qualified financial advisor before making any investment decisions. Trading stocks involves risk, including the potential loss of principal.</p>
    </div>
    """, unsafe_allow_html=True)

# Function to generate prediction verdict
def generate_prediction_verdict(last_price, ml_results):
    verdict = "Analysis Pending..."
    icon = "🤔"
    border_color = "gray"
    advice = "Load data and run ML pipeline to get predictions."

    if not ml_results:
        return f"""
        <div class='verdict-box' style='border-left-color:{border_color};'>
             <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <span class='verdict-icon'>{icon}</span><span class='verdict-title' style='margin-bottom:0;'>Prediction Verdict: {verdict}</span>
            </div>
            <p class='verdict-text'>{advice}</p>
        </div>
        """

    try:
        pred_reg = ml_results.get('regression', {})
        pred_cls = ml_results.get('classification', {})

        next_price = pred_reg.get('prediction', None)
        trend = pred_cls.get('trend', 'N/A')
        probability = pred_cls.get('probability', 0.5) # Default 0.5

        if next_price is None or trend == 'N/A' or probability is None:
            verdict = "Incomplete Predictions"
            advice = "One or both ML models failed to produce a prediction or probability. Check ML pipeline steps and data quality."
            icon = "❓"
            border_color = "orange"
        else:
            # Use .item() if numpy float
            if isinstance(next_price, np.number): next_price = next_price.item()
            if isinstance(probability, np.number): probability = probability.item()

            pct_change = ((next_price - last_price) / last_price) * 100 if last_price else 0
            confidence = abs(probability - 0.5) * 2 # Scale probability to 0-1 confidence range centered at 0.5

            # Define verdict based on combined signals
            if trend == "UP":
                border_color = NEON_GREEN # Default UP color
                if pct_change > 1.0 and confidence > 0.5: # Price up significantly, reasonably confident UP trend
                    verdict = "Strong Bullish Signal"
                    icon = "🚀"
                    advice = f"Both models suggest a strong upward move ({pct_change:+.1f}%) with good confidence ({probability:.1%}). Potential buying opportunity, but monitor closely."
                elif pct_change > 0: # Price up moderately or lower confidence UP trend
                    verdict = "Cautiously Bullish Signal"
                    icon = "📈"
                    advice = f"Models lean positive ({pct_change:+.1f}%, {probability:.1%} UP conf.). Consider upside potential, but be aware of moderate price target or confidence."
                else: # Trend UP but Price prediction DOWN/Flat
                    verdict = "Mixed Signal (Bullish Trend Conflict)"
                    icon = "☯️"
                    border_color = NEON_PINK # Use neutral color for conflict
                    advice = f"Trend model predicts UP ({probability:.1%} conf.), but price model forecasts down/flat ({pct_change:+.1f}%). High uncertainty. Wait for clearer signals."
            elif trend == "DOWN":
                border_color = "red" # Default DOWN color
                if pct_change < -1.0 and confidence > 0.5: # Price down significantly, reasonably confident DOWN trend
                    verdict = "Strong Bearish Signal"
                    icon = "💥"
                    advice = f"Both models suggest a strong downward move ({pct_change:+.1f}%) with good confidence ({probability:.1%} DOWN). Potential shorting opportunity or time to exit long positions."
                elif pct_change < 0: # Price down moderately or lower confidence DOWN trend
                    verdict = "Cautiously Bearish Signal"
                    icon = "📉"
                    advice = f"Models lean negative ({pct_change:+.1f}%, {probability:.1%} DOWN conf.). Consider downside risk, but be aware of moderate price target or confidence."
                else: # Trend DOWN but Price prediction UP/Flat
                    verdict = "Mixed Signal (Bearish Trend Conflict)"
                    icon = "☯️"
                    border_color = NEON_PINK # Use neutral color for conflict
                    advice = f"Trend model predicts DOWN ({probability:.1%} conf.), but price model forecasts up/flat ({pct_change:+.1f}%). High uncertainty. Wait for clearer signals."
            else: # Should not happen if trend is UP/DOWN, but as fallback
                verdict = "Uncertain Outcome"
                icon = "❓"
                border_color = "gray"
                advice = "Trend prediction is unavailable or inconclusive. Cannot form a combined verdict."

    except Exception as e:
        verdict = "Verdict Calculation Error"
        icon = "⚠️"
        border_color = "orange"
        advice = f"An error occurred while generating the verdict: {str(e)}"
        print(traceback.format_exc()) # Log error for debugging

    # Ensure border_color is a valid CSS color string
    valid_colors = [NEON_PINK, NEON_GREEN, "red", "gray", "orange"]
    if border_color not in valid_colors:
        border_color = "gray" # Fallback to gray

    return f"""
    <div class='verdict-box gradient-border' style='border-left: 5px solid {border_color};'>
         <div style='display: flex; align-items: center; margin-bottom: 10px;'>
            <span class='verdict-icon'>{icon}</span><span class='verdict-title' style='margin-bottom:0;'>Prediction Verdict: {verdict}</span>
        </div>
        <p class='verdict-text'>{advice}</p>
    </div>
    """

# Function to create prediction summary for download
def create_prediction_summary(ticker, last_price, ml_results):
    summary = {"Ticker": ticker, "Last Close Price": f"{last_price:.2f}", "Prediction Target Date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")}
    if ml_results:
        pred_reg = ml_results.get('regression', {})
        pred_cls = ml_results.get('classification', {})

        next_price = pred_reg.get('prediction')
        lower_bound = pred_reg.get('lower_bound')
        upper_bound = pred_reg.get('upper_bound')
        trend = pred_cls.get('trend')
        probability = pred_cls.get('probability') # Probability of UP trend (class 1)

        # Use .item() if numpy float and handle None
        summary["Predicted Price (5d)"] = f"{next_price.item():.2f}" if isinstance(next_price, np.number) else f"{next_price:.2f}" if next_price is not None else "N/A"
        summary["Price Lower Bound (95% CI)"] = f"{lower_bound.item():.2f}" if isinstance(lower_bound, np.number) else f"{lower_bound:.2f}" if lower_bound is not None else "N/A"
        summary["Price Upper Bound (95% CI)"] = f"{upper_bound.item():.2f}" if isinstance(upper_bound, np.number) else f"{upper_bound:.2f}" if upper_bound is not None else "N/A"
        summary["Predicted Trend (5d)"] = trend if trend is not None else "N/A"
        summary["Trend Probability (Up)"] = f"{probability.item():.3f}" if isinstance(probability, np.number) else f"{probability:.3f}" if probability is not None else "N/A"
    else:
        summary["Predicted Price (5d)"] = "N/A"
        summary["Price Lower Bound (95% CI)"] = "N/A"
        summary["Price Upper Bound (95% CI)"] = "N/A"
        summary["Predicted Trend (5d)"] = "N/A"
        summary["Trend Probability (Up)"] = "N/A"

    return summary

# --- Main Application Logic ---
def main():
    # --- Initialize Session State ---
    if 'sidebar_quote' not in st.session_state:
        st.session_state.sidebar_quote = "Click a symbol above for a message..."
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'info' not in st.session_state:
        st.session_state.info = None
    if 'ticker' not in st.session_state:
        st.session_state.ticker = "AAPL" # Default ticker
    if 'error' not in st.session_state:
        st.session_state.error = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Linear Regression (Price Prediction)" # Default ML selection
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = None
    if 'corr_df' not in st.session_state:
        st.session_state.corr_df = None
    if 'ml_error' not in st.session_state:
        st.session_state.ml_error = None
    # Add state for colorscales
    if 'selected_diverging_colorscale' not in st.session_state:
        st.session_state.selected_diverging_colorscale = 'RdBu'
    if 'selected_coefficient_colorscale' not in st.session_state:
        st.session_state.selected_coefficient_colorscale = 'Picnic'
    if 'selected_period' not in st.session_state:
         st.session_state.selected_period = "1 Year" # Default period

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;' class='gradient-border'>
            <h2 style='color:{NEON_PINK}; text-shadow: 0 0 5px {NEON_PINK}; margin-bottom: 5px;'>
                SquidStock Controls
            </h2>
            <p style='color:white; font-size:0.9em; margin-top:0;'>Configure your analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # --- Sidebar Symbol Buttons for Quotes ---
        st.markdown("<h4 style='color:white; text-align:center; margin-top: 20px;'>Seek Wisdom?</h4>", unsafe_allow_html=True)
        quotes = {
            "⭕": "“In this game called life, we are not competitors but teammates. Stick together.” – Player 067",
            "△": "“I don't trust people. Especially those who get themselves into debt.” – Player 218 (Cho Sang-woo)",
            "□": "“Everyone is equal while they play this game. Here, every player gets to play a fair game under the same conditions.” – The Front Man",
            "☂": "“You don't trust people because they are trustworthy. You do it because you have nothing else to rely on.” – Player 001 (Oh Il-nam)"
        }
        cols = st.columns(4)
        symbols = ["⭕", "△", "□", "☂"]
        for i, symbol in enumerate(symbols):
            # Use custom class for styling these specific buttons if needed
            if cols[i].button(symbol, key=f"quote_btn_{symbol}", help=f"Click for a {symbol} quote"): # Removed symbol-button class for simplicity now
                st.session_state.sidebar_quote = quotes[symbol]

        # Display the selected quote
        st.markdown(f"<div class='quote-box'>{st.session_state.sidebar_quote}</div>", unsafe_allow_html=True)
        st.markdown("---") # Separator

        # --- Data Source Selection ---
        st.markdown(f"""
        <div class='hover-card pulse' style='text-align:center;'>
            <h4 style='color:{NEON_PINK}; margin:0;'>SELECT DATA SOURCE</h4>
        </div>
        """, unsafe_allow_html=True)

        data_source = st.radio(
            "Choose your data source",
            ["Yahoo Finance API", "Upload CSV File"],
            label_visibility="collapsed", # Hide the label itself
            key="data_source_radio",
            horizontal=True, # Display options side-by-side
        )

        # --- Conditional Input based on Data Source ---
        ticker_input = st.session_state.ticker # Use session state value by default
        period_input = st.session_state.selected_period # Use selected period

        if data_source == "Yahoo Finance API":
            ticker_input = st.text_input(
                "Enter Stock Ticker Symbol",
                value=st.session_state.ticker, # Use session state value
                key="ticker_input_api",
                help="e.g., AAPL, MSFT, GOOGL"
            ).upper()

            period_options = ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Max"]
            default_period_index = period_options.index(st.session_state.selected_period) if st.session_state.selected_period in period_options else period_options.index("1 Year")

            period_input_select = st.selectbox(
                "Select Time Period",
                period_options,
                index=default_period_index,
                key="period_select"
            )
            # Update session state *only if changed* to avoid unnecessary resets
            if period_input_select != st.session_state.selected_period:
                 st.session_state.selected_period = period_input_select
                 # Optionally reset data/ML here if period change should force reload,
                 # but the Load button handles this explicitly now.


            # Fetch/Refresh Button for API
            if st.button("Load / Refresh API Data", key="load_api_data"):
                # Reset ML results when loading new data
                st.session_state.ml_results = None
                st.session_state.trained_models = None
                st.session_state.corr_df = None
                st.session_state.ml_error = None
                period_map = {"3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"}
                with st.spinner(f"Fetching {ticker_input} data for {st.session_state.selected_period}..."):
                    data, info, error = load_stock_data(ticker_input, period=period_map[st.session_state.selected_period])
                    st.session_state.data = data
                    st.session_state.info = info
                    st.session_state.ticker = ticker_input # Update ticker in state
                    st.session_state.error = error
                    if error:
                        st.error(error) # Show error message immediately in sidebar
                    elif data is not None and not data.empty:
                        st.success(f"Loaded {len(data)} rows for {ticker_input}")
                    else:
                         st.warning(f"No data loaded for {ticker_input}. API returned empty or error occurred.")
                         st.session_state.data = None # Ensure data is None if loading failed


            st.markdown(f"""
            <div class='hover-card' style='margin-top:10px;'>
                <p style='font-size:0.9em; text-align:center; margin:0;'>
                    Need a ticker? <a href='https://finance.yahoo.com/lookup' target='_blank' style='color:{NEON_GREEN};'>Look one up here</a>
                </p>
            </div>
            """, unsafe_allow_html=True)

        else: # CSV Upload option
            st.markdown(f"""
            <div class='hover-card'>
                <h4 style='color:{NEON_PINK};'>Upload Your CSV File</h4>
                <p style='font-size:0.8em; margin-bottom: 5px;'>Requires: Date (index or column), Open, High, Low, Close, Volume.</p>
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Choose a CSV file", type="csv", key="csv_uploader",
                help="Upload a CSV with Date, OHLC, Volume columns"
            )

            # Template Download and Format Help
            example_csv = pd.DataFrame({
                'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
                'Open': [100, 101, 102], 'High': [102, 103, 104], 'Low': [99, 100, 101],
                'Close': [101, 102, 103], 'Volume': [10000, 12000, 11000]
            }).set_index('Date')
            csv_buffer = io.StringIO()
            example_csv.to_csv(csv_buffer)
            st.download_button(
                label="Download Example CSV",
                data=csv_buffer.getvalue(),
                file_name="squidstock_example.csv",
                mime="text/csv",
                key="download_csv_template"
            )
            with st.expander("CSV Format Details"):
                st.markdown("""
                - **Date Column:** Must be present. Can be the index or a column named 'Date', 'Time', 'Timestamp', etc. It should be parseable as dates (e.g., YYYY-MM-DD).
                - **Required OHLCV Columns:** `Open`, `High`, `Low`, `Close`, `Volume`. Case-insensitive matching is attempted. Column order doesn't matter.
                - **Optional Columns:** Other columns will be ignored.
                - **Data Order:** Data should ideally be sorted chronologically (oldest first), but the app will attempt to sort it by date.
                - **Data Quality:** Ensure columns contain numeric data. Missing values might cause issues if not handled correctly by the cleaning process.
                """, unsafe_allow_html=True)


            if uploaded_file is not None:
                 # Reset ML results when loading new data
                 st.session_state.ml_results = None
                 st.session_state.trained_models = None
                 st.session_state.corr_df = None
                 st.session_state.ml_error = None
                 with st.spinner("Processing CSV file..."):
                     try:
                         # Read CSV
                         uploaded_data = pd.read_csv(uploaded_file)
                         processed_data = None
                         error_msg = None

                         if uploaded_data.empty:
                             error_msg = "Uploaded CSV file is empty."
                         else:
                             # --- Date Column Processing ---
                             potential_date_cols = [col for col in uploaded_data.columns if 'date' in col.lower() or 'time' in col.lower()]
                             date_col_found = False
                             date_col = None # Initialize date_col

                             # Try potential date columns first
                             if potential_date_cols:
                                 for col in potential_date_cols:
                                     try:
                                         # Try parsing with infer_datetime_format for speed and flexibility
                                         test_parse = pd.to_datetime(uploaded_data[col], infer_datetime_format=True, errors='coerce')
                                         # Check if parsing was successful for most values
                                         if test_parse.notna().mean() > 0.8: # Heuristic: if >80% parse ok
                                             uploaded_data[col] = test_parse
                                             uploaded_data.set_index(col, inplace=True)
                                             date_col_found = True
                                             date_col = col # Store the column name used as index
                                             st.info(f"Used column '{col}' as Date index.")
                                             break # Stop after finding the first valid date column
                                         else:
                                             continue
                                     except Exception:
                                         continue # Try the next potential date column

                             # If no column worked, try parsing the index
                             if not date_col_found:
                                 try:
                                     # Check if index is already DatetimeIndex
                                     if not isinstance(uploaded_data.index, pd.DatetimeIndex):
                                         test_parse_index = pd.to_datetime(uploaded_data.index, infer_datetime_format=True, errors='coerce')
                                         if test_parse_index.notna().mean() > 0.8:
                                             uploaded_data.index = test_parse_index
                                             date_col_found = True
                                             st.info("Used existing index as Date index.")
                                         else:
                                             error_msg = "Could not reliably parse dates from existing index."
                                     else: # Index is already datetime
                                        date_col_found = True
                                        st.info("Using existing DatetimeIndex.")

                                 except Exception as e:
                                      error_msg = f"Could not automatically parse dates from columns or index. Ensure a valid date column/index exists. Error: {e}"

                             # --- OHLCV Column Processing ---
                             if date_col_found and error_msg is None:
                                 # Find OHLCV columns (case-insensitive)
                                 required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                                 col_map = {}
                                 missing_cols = []
                                 current_cols = list(uploaded_data.columns)
                                 # if date_col: # If a column was used as index, it's no longer in columns (already handled by list(cols))
                                 #      pass

                                 for req_col in required_cols:
                                     found = False
                                     for actual_col in current_cols:
                                         if req_col.lower() == actual_col.lower():
                                             col_map[req_col] = actual_col
                                             found = True
                                             break
                                     if not found:
                                         missing_cols.append(req_col)

                                 if missing_cols:
                                     error_msg = f"CSV missing required columns: {', '.join(missing_cols)}. Required: Open, High, Low, Close, Volume (case-insensitive)."
                                 else:
                                     # Select and rename columns
                                     processed_data = uploaded_data[[col_map[col] for col in required_cols]].copy()
                                     processed_data.columns = required_cols # Standardize column names

                                     # Ensure data is numeric and handle potential errors
                                     for col in required_cols:
                                         processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

                                     # Check for NaNs introduced by coercion
                                     if processed_data.isnull().any().any():
                                         nan_cols = processed_data.columns[processed_data.isnull().any()].tolist()
                                         st.warning(f"Non-numeric values found and converted to NaN in columns: {nan_cols}. Attempting to clean.")
                                         # Attempt basic cleaning (e.g., ffill), more robust cleaning might be needed
                                         processed_data.ffill(inplace=True)
                                         processed_data.bfill(inplace=True)
                                         # Drop rows if still NaN after filling, especially in 'Close'
                                         processed_data.dropna(subset=['Close'], inplace=True)


                                     if processed_data.empty:
                                          error_msg = "Data became empty after handling non-numeric values. Please check CSV quality."
                                          processed_data = None
                                     elif 'Volume' in processed_data.columns:
                                          # Ensure Volume is integer type after cleaning
                                          processed_data['Volume'] = processed_data['Volume'].fillna(0).astype(np.int64)

                                     # Sort by date index
                                     if processed_data is not None:
                                          # Drop rows with NaT index if parsing failed partially
                                          processed_data.dropna(axis=0, how='all', subset=processed_data.columns, inplace=True) # Drop rows if all values NaN
                                          processed_data = processed_data[processed_data.index.notna()]
                                          if processed_data.empty:
                                              error_msg = "Data became empty after removing rows with invalid dates."
                                              processed_data = None
                                          else:
                                              processed_data.sort_index(inplace=True)


                         # Update session state based on outcome
                         if error_msg:
                             st.error(error_msg)
                             st.session_state.data = None
                             st.session_state.info = None
                             st.session_state.ticker = "CSV Error"
                             st.session_state.error = error_msg
                         elif processed_data is not None and not processed_data.empty:
                             st.session_state.data = processed_data
                             # Create dummy info for CSV
                             st.session_state.info = {
                                 'longName': f'CSV Data ({uploaded_file.name})',
                                 'symbol': 'CSV Upload', 'sector': 'N/A', 'industry': 'N/A',
                                 'marketCap': None, 'trailingPE': None, 'website': '#',
                                 'longBusinessSummary': f'Data loaded from user-provided CSV file: {uploaded_file.name}. Contains {len(processed_data)} rows from {processed_data.index.min().date()} to {processed_data.index.max().date()}.'
                             }
                             st.session_state.ticker = "CSV Data"
                             st.session_state.error = None
                             st.success(f"CSV file '{uploaded_file.name}' loaded successfully ({len(processed_data)} rows)!")
                         else:
                             # Catch any other case where processed_data is None or empty without an error message
                             st.error("Failed to process CSV data. Check file format and content, especially dates and numeric values.")
                             st.session_state.data = None
                             st.session_state.info = None
                             st.session_state.ticker = "CSV Error"
                             st.session_state.error = "Failed to process CSV data."


                     except Exception as e:
                         st.error(f"Error processing CSV: {str(e)}")
                         st.error(traceback.format_exc())
                         st.session_state.data = None
                         st.session_state.info = None
                         st.session_state.ticker = "CSV Error"
                         st.session_state.error = f"Error processing CSV: {str(e)}"

        # --- Disclaimer ---
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"""
        <div class='gradient-border' style='margin-top:15px;'>
            <p style='color:{NEON_PINK}; font-weight:bold; margin-bottom:5px;'>⚠️ Important Note</p>
            <p style='color:white; font-size:0.8em; margin:0;'>This tool is for educational/entertainment purposes. Market predictions are uncertain. Not financial advice.</p>
        </div>
        """, unsafe_allow_html=True)

        # --- Footer ---
        st.sidebar.markdown(f"""
        <div style='text-align:center; margin-top:30px; opacity:0.7;'>
            <p style='font-size:0.7em; color:gray;'>© {datetime.now().year} SquidStock Survival</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Main Page Display ---
    squid_header()

    # Check if data is loaded and valid before showing tabs
    if st.session_state.data is None or st.session_state.data.empty:
        # If there was an error during loading, show it prominently
        if st.session_state.error:
             st.error(f"Data Loading Failed: {st.session_state.error}")
             st.info("Please check the ticker symbol/CSV file and try again, or select a different data source/period.")
        # Otherwise, show the welcome screen
        else:
            show_welcome()

    else:
        # Data is loaded, proceed with tabs
        data = st.session_state.data
        info = st.session_state.info
        ticker = st.session_state.ticker # Get ticker (could be symbol or "CSV Data")
        current_price = data['Close'].iloc[-1] # Needed for verdict and download

        # Display Company Info (from API or dummy for CSV)
        if info:
             display_company_info(info)
        else:
             st.sidebar.info("No detailed company information available for this data source.")


        st.markdown(f"""
        <div class='dashboard-section gradient-border' style='margin-bottom: 5px;'>
            <h2 style='color:{NEON_PINK}; text-align:center; margin-bottom:5px; text-shadow: 0 0 5px {NEON_PINK};'>
                Analysis for: {st.session_state.ticker}
            </h2>

        """, unsafe_allow_html=True)

        # Create tabs
        tab1, tab2 = st.tabs(["📊 Stock Analysis", "🎮 ML Predictions & Insights"])

        # --- Tab 1: Stock Analysis ---
        with tab1:
            if data is None or data.empty:
                 st.warning("No data available to display analysis.")
            else:
                # --- View Raw Data ---
                with st.expander("View Raw Data", expanded=False):
                    st.dataframe(data.head(1000), use_container_width=True) # Show max 1000 rows for performance
                    st.caption(f"Displaying first {min(len(data), 1000)} of {len(data)} rows of historical data.")
                    # Optional: Download full raw data
                    raw_csv = data.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Raw Data (CSV)",
                        data=raw_csv,
                        file_name=f"{ticker}_raw_data_{data.index.min():%Y%m%d}_{data.index.max():%Y%m%d}.csv",
                        mime="text/csv",
                        key="download_raw_data"
                    )


                st.markdown("---") # Separator

                # --- Key Metrics Row ---
                st.markdown("#### Current Snapshot", unsafe_allow_html=True)
                # current_price already defined outside tabs
                prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                price_change = current_price - prev_close
                pct_change = (price_change / prev_close * 100) if prev_close and prev_close != 0 else 0
                daily_high = data['High'].iloc[-1]
                daily_low = data['Low'].iloc[-1]
                volume = data['Volume'].iloc[-1]
                # Calculate avg_volume safely, handle potential NaNs or short series
                vol_roll = data['Volume'].rolling(window=20, min_periods=5).mean()
                avg_volume = vol_roll.iloc[-1] if not vol_roll.empty and not pd.isna(vol_roll.iloc[-1]) else None

                # Use the metrics card styling
                st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                price_color = NEON_GREEN if price_change >= 0 else "red"
                price_icon = "📈" if price_change >= 0 else "📉"
                st.markdown(f"""
                <div class="metric-card {'up' if price_change >= 0 else 'down'}">
                    <div class="metric-title">Last Price</div>
                    <div class="metric-value" style="color:{price_color}">{price_icon} ${current_price:.2f}</div>
                    <div class="metric-subtext" style="color:{price_color}">{price_change:+.2f} ({pct_change:+.2f}%)</div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-card neutral">
                    <div class="metric-title">Day High</div>
                    <div class="metric-value" style="color:white">${daily_high:.2f}</div>
                    <div class="metric-subtext">Latest Day's Peak</div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-card neutral">
                    <div class="metric-title">Day Low</div>
                    <div class="metric-value" style="color:white">${daily_low:.2f}</div>
                    <div class="metric-subtext">Latest Day's Bottom</div>
                </div>""", unsafe_allow_html=True)

                vol_ratio_text = "N/A"
                vol_color = "white"
                vol_class = "neutral"
                if avg_volume is not None and avg_volume != 0:
                    vol_ratio = (volume / avg_volume)
                    vol_ratio_text = f"{vol_ratio:.2f}x vs 20D Avg"
                    vol_color = NEON_GREEN if vol_ratio > 1.1 else "red" if vol_ratio < 0.9 else "white"
                    vol_class = "up" if vol_ratio > 1.1 else "down" if vol_ratio < 0.9 else "neutral"
                else:
                    vol_ratio_text = "Avg Vol N/A"

                st.markdown(f"""
                 <div class="metric-card {vol_class}">
                    <div class="metric-title">Volume</div>
                    <div class="metric-value" style="color:{vol_color}">{volume:,.0f}</div>
                    <div class="metric-subtext" style="color:{vol_color if vol_class != 'neutral' else 'gray'};">{vol_ratio_text}</div>
                </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True) # Close metrics-container

                # --- Charts ---
                st.markdown("---")
                st.markdown(f"""
                <div class='dashboard-section'>
                    <h3 style='color:{NEON_PINK}; margin-bottom: 5px;'>Price Chart & RSI</h3>
                    <p style='font-size:0.9em; margin-top:0;'>Candlestick price action with Moving Averages and Relative Strength Index (RSI).</p>
                </div>
                """, unsafe_allow_html=True)
                try:
                    fig_price, fig_volume = display_stock_charts(data, ticker)
                    st.plotly_chart(fig_price, use_container_width=True)

                    st.markdown(f"""
                    <div class='dashboard-section'>
                        <h3 style='color:{NEON_PINK}; margin-bottom: 5px;'>Volume Analysis</h3>
                         <p style='font-size:0.9em; margin-top:0;'>Trading volume bars and 20-day moving average.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(fig_volume, use_container_width=True)
                except Exception as chart_error:
                    st.error(f"Error generating charts: {chart_error}")
                    st.info("This might be due to issues in the loaded data.")
                    st.error(traceback.format_exc())


                # --- Key Statistics ---
                st.markdown("---")
                st.markdown("#### Period Statistics", unsafe_allow_html=True)
                st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)

                period_high = data['High'].max()
                period_low = data['Low'].min()
                period_avg_volume = data['Volume'].mean()
                # Calculate annualized volatility for the period
                daily_returns = data['Close'].pct_change().dropna() # Drop first NaN
                period_volatility = daily_returns.std() * np.sqrt(252) * 100 if not daily_returns.empty else 0 # Annualized %

                st.markdown(f"""
                <div class="metric-card neutral">
                    <div class="metric-title">Period High</div>
                    <div class="metric-value" style="color:white">${period_high:.2f}</div>
                    <div class="metric-subtext">{data.index.min():%b %d, %Y} - {data.index.max():%b %d, %Y}</div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                 <div class="metric-card neutral">
                    <div class="metric-title">Period Low</div>
                    <div class="metric-value" style="color:white">${period_low:.2f}</div>
                     <div class="metric-subtext">{data.index.min():%b %d, %Y} - {data.index.max():%b %d, %Y}</div>
                 </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-card neutral">
                    <div class="metric-title">Avg Daily Volume</div>
                    <div class="metric-value" style="color:white">{period_avg_volume:,.0f}</div>
                    <div class="metric-subtext">For the selected period</div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-card neutral">
                    <div class="metric-title">Annualized Volatility</div>
                    <div class="metric-value" style="color:white">{period_volatility:.2f}%</div>
                    <div class="metric-subtext">Based on daily returns</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True) # Close metrics-container

        # --- Tab 2: ML Predictions & Insights ---
        with tab2:
            st.markdown(f"""

            """, unsafe_allow_html=True)

            # --- ML Model Selection ---
            st.markdown(f"""
            <div class='hover-card pulse'>
                <h4 style='color:{NEON_PINK};'>SELECT ML MODEL FOCUS</h4>
                <p style='font-size:0.9em;'>Choose which prediction type to focus on:</p>
            </div>
            """, unsafe_allow_html=True)

            model_options = ["Linear Regression (Price Prediction)", "Logistic Regression (Trend Prediction)"]
            # Use session state to keep selection consistent
            selected_model = st.selectbox(
                "Select ML Model Focus",
                model_options,
                index=model_options.index(st.session_state.selected_model), # Use state for index
                key="ml_model_select_main",
                label_visibility="collapsed" # Hide label as header is descriptive
            )
            # Update session state when selection changes
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                st.rerun() # Rerun to update the display based on selection


            st.markdown("---") # Separator

            # --- Interactive ML Pipeline ---
            st.markdown(f"""
            <div class='dashboard-section'>
                <h3 style='color:{NEON_PINK}; text-align:center;'>ML Pipeline Visualization</h3>

            """, unsafe_allow_html=True)

            # Define pipeline steps
            ml_steps = ["Load Data", "Preprocessing", "Feature Engineering", "Train/Test Split", "Model Training", "Evaluation", "Prediction Ready"]
            pipeline_cols = st.columns(len(ml_steps))
            step_status_msgs = {} # To hold status messages below icons

            # Initialize progress bar and status text
            progress_bar = st.progress(0)
            status_text_area = st.empty() # Placeholder for overall status updates

            # Check if ML results are already in session state
            run_ml_pipeline = False
            # Run if no results and no previous error for this data
            if st.session_state.ml_results is None and st.session_state.ml_error is None:
                 run_ml_pipeline = True
            # Also allow re-running
            if st.button("Re-run ML Pipeline", key="rerun_ml"):
                 run_ml_pipeline = True
                 # Reset states before running again
                 st.session_state.ml_results = None
                 st.session_state.trained_models = None
                 st.session_state.corr_df = None
                 st.session_state.ml_error = None


            if run_ml_pipeline:
                with st.spinner("Running ML Pipeline... Please Wait."):
                    try:
                        # --- Step 1: Load Data ---
                        status_text_area.markdown(f"<p style='color:{NEON_GREEN};'>Step 1: Accessing loaded data...</p>", unsafe_allow_html=True)
                        if data is None or data.empty:
                            raise ValueError("Input data is not available for ML pipeline.")
                        with pipeline_cols[0]:
                            st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps[0]}</p><p class='status-icon'>💾</p><p class='status-message'>Data loaded:<br>{data.shape[0]} rows</p></div>", unsafe_allow_html=True)
                        progress_bar.progress(1/len(ml_steps))
                        time.sleep(0.05) # Small delay for visual flow

                        # --- Step 2 & 3: Preprocessing & Feature Engineering ---
                        status_text_area.markdown(f"<p style='color:{NEON_GREEN};'>Step 2 & 3: Preprocessing & Feature Engineering...</p>", unsafe_allow_html=True)
                        # Use session state vars directly
                        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, latest_data_scaled, corr_df_result = prepare_ml_data(data)

                        # Check result of prepare_ml_data
                        if isinstance(corr_df_result, str): # If the last element is an error string
                            st.session_state.ml_error = corr_df_result # Capture the error message
                            raise ValueError(st.session_state.ml_error)
                        if X_train is None or corr_df_result is None: # General check if preparation failed
                             raise ValueError("Data preparation failed. Check data quality or period length.")
                        st.session_state.corr_df = corr_df_result # Store successful result

                        with pipeline_cols[1]:
                            st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps[1]}</p><p class='status-icon'>🧹</p><p class='status-message'>NaNs handled<br>Data Scaled</p></div>", unsafe_allow_html=True)
                        progress_bar.progress(2/len(ml_steps))
                        time.sleep(0.05)

                        with pipeline_cols[2]:
                             feature_count = X_train.shape[1] if X_train is not None else 0
                             st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps[2]}</p><p class='status-icon'>🛠️</p><p class='status-message'>Features created:<br>MAs, RSI, Vol, etc. ({feature_count} total)</p></div>", unsafe_allow_html=True)
                        progress_bar.progress(3/len(ml_steps))
                        time.sleep(0.05)

                        # --- Step 4: Train/Test Split ---
                        status_text_area.markdown(f"<p style='color:{NEON_GREEN};'>Step 4: Splitting data...</p>", unsafe_allow_html=True)
                        train_samples = len(X_train) if X_train is not None else 0
                        test_samples = len(X_test) if X_test is not None else 0
                        with pipeline_cols[3]:
                             st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps[3]}</p><p class='status-icon'>✂️</p><p class='status-message'>Train: {train_samples} samples<br>Test: {test_samples} samples</p></div>", unsafe_allow_html=True)
                        progress_bar.progress(4/len(ml_steps))
                        time.sleep(0.05)

                        # --- Step 5: Model Training ---
                        status_text_area.markdown(f"<p style='color:{NEON_GREEN};'>Step 5: Training models...</p>", unsafe_allow_html=True)
                        ml_results_run, trained_models_run, ml_error_run = train_and_predict(X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, latest_data_scaled)

                        if ml_error_run:
                            st.session_state.ml_error = ml_error_run
                            raise ValueError(f"Model training failed: {st.session_state.ml_error}")
                        if not ml_results_run:
                             raise ValueError("Model training did not produce results.")
                        st.session_state.ml_results = ml_results_run # Store successful results
                        st.session_state.trained_models = trained_models_run

                        with pipeline_cols[4]:
                             st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps[4]}</p><p class='status-icon'>🧠</p><p class='status-message'>LinReg Trained<br>LogReg Trained</p></div>", unsafe_allow_html=True)
                        progress_bar.progress(5/len(ml_steps))
                        time.sleep(0.05)

                        # --- Step 6: Evaluation ---
                        status_text_area.markdown(f"<p style='color:{NEON_GREEN};'>Step 6: Evaluating performance...</p>", unsafe_allow_html=True)
                        reg_rmse = st.session_state.ml_results.get('regression', {}).get('rmse', float('nan'))
                        cls_acc_val = st.session_state.ml_results.get('classification', {}).get('accuracy') # Get accuracy, could be None
                        cls_acc = cls_acc_val * 100 if cls_acc_val is not None else float('nan') # Convert to % if not None
                        rmse_text = f"${reg_rmse:.2f}" if not np.isnan(reg_rmse) else "N/A"
                        acc_text = f"{cls_acc:.1f}%" if not np.isnan(cls_acc) else "N/A"
                        with pipeline_cols[5]:
                             st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps[5]}</p><p class='status-icon'>📊</p><p class='status-message'>Price RMSE: {rmse_text}<br>Trend Acc: {acc_text}</p></div>", unsafe_allow_html=True)
                        progress_bar.progress(6/len(ml_steps))
                        time.sleep(0.05)

                        # --- Step 7: Prediction Ready ---
                        status_text_area.markdown(f"<p style='color:{NEON_GREEN};'>Step 7: Predictions generated!</p>", unsafe_allow_html=True)
                        with pipeline_cols[6]:
                             st.markdown(f"<div class='ml-step success'><p class='step-title'>{ml_steps[6]}</p><p class='status-icon'>✨</p><p class='status-message'>Analysis complete<br>Results below</p></div>", unsafe_allow_html=True)
                        progress_bar.progress(1.0)

                        status_text_area.success("✅ ML Pipeline Completed Successfully!")
                        st.session_state.ml_error = None # Clear any previous error on success

                    except Exception as e:
                        st.session_state.ml_error = f"Pipeline failed: {str(e)}"
                        status_text_area.error(f"❌ ML Pipeline Failed: {str(e)}")
                        st.error(traceback.format_exc()) # Show traceback in main area
                        # Clear results if pipeline fails
                        st.session_state.ml_results = None
                        st.session_state.trained_models = None
                        st.session_state.corr_df = None
                        # Visually mark incomplete steps (optional, simple approach: clear progress)
                        progress_bar.progress(0)
            else: # ML was not run, display current status
                if st.session_state.ml_error:
                    status_text_area.error(f"❌ ML Pipeline Previously Failed: {st.session_state.ml_error}")
                elif st.session_state.ml_results:
                    status_text_area.success("✅ ML Pipeline Previously Completed Successfully.")
                    # Optionally reconstruct the visual pipeline state here if desired
                    progress_bar.progress(1.0)
                    # Display minimal pipeline state
                    for i, step in enumerate(ml_steps):
                         with pipeline_cols[i]:
                              st.markdown(f"<div class='ml-step success'><p class='step-title'>{step}</p><p class='status-icon'>✔️</p><p class='status-message'>Completed</p></div>", unsafe_allow_html=True)

                else:
                    status_text_area.info("ML Pipeline not yet run for this data. Click 'Re-run ML Pipeline'.")
                    progress_bar.progress(0)


            # --- Display ML Results Area ---
            st.markdown("---")

            # Display results OR error message OR prompt to run
            if st.session_state.ml_results:
                # --- Prediction Verdict ---
                try:
                     verdict_html = generate_prediction_verdict(current_price, st.session_state.ml_results)
                     st.markdown(verdict_html, unsafe_allow_html=True)
                except Exception as verdict_err:
                     st.error(f"Error displaying prediction verdict: {verdict_err}")
                     st.error(traceback.format_exc())


                # --- Download Predictions ---
                try:
                    prediction_summary_dict = create_prediction_summary(ticker, current_price, st.session_state.ml_results)
                    # Convert dict to DataFrame for CSV download
                    pred_df = pd.DataFrame([prediction_summary_dict])
                    pred_csv = pred_df.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        label="📥 Download Prediction Summary (CSV)",
                        data=pred_csv,
                        file_name=f"{ticker}_prediction_summary_{datetime.now():%Y%m%d}.csv",
                        mime="text/csv",
                        key="download_prediction_csv"
                    )
                except Exception as download_err:
                    st.warning(f"Could not prepare prediction download: {download_err}")

                st.markdown("---")


                # --- Display based on selected model focus ---
                if "Linear Regression" in st.session_state.selected_model:
                    st.markdown(f"""
                    <div class='dashboard-section gradient-border'>
                        <h3 style='color:{NEON_PINK}; text-align:center;'>Price Prediction Results (Linear Regression)</h3>
                    """, unsafe_allow_html=True)
                    # --- Linear Regression Description ---
                    st.markdown(f"""
                    <div class='hover-card' style='background-color:rgba(0,0,0,0.7); border-left:4px solid {NEON_GREEN}; margin-bottom:10px;'>
                        <b>What is Linear Regression?</b><br>
                        Linear regression is a statistical method that models the relationship between a target variable and one or more features by fitting a straight line. In this app, it predicts the <b>future stock price </b> based on historical price and technical indicators.
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        fig_pred_price = display_price_prediction(current_price, st.session_state.ml_results, ticker)
                        st.plotly_chart(fig_pred_price, use_container_width=True)
                    except Exception as plot_err:
                        st.error(f"Error displaying price prediction plot: {plot_err}")
                        st.error(traceback.format_exc())

                    # Additional Regression Plots (only for Linear Regression)
                    with st.expander("Regression Model Performance Plots"):
                         reg_data = st.session_state.ml_results.get('regression')
                         if reg_data and 'y_true' in reg_data and 'y_pred' in reg_data and reg_data['y_true'] is not None and reg_data['y_pred'] is not None:
                            try:
                                plot_df = pd.DataFrame({'Actual': reg_data['y_true'], 'Predicted': reg_data['y_pred'], 'Residuals': reg_data['residuals']})

                                col_p1, col_p2 = st.columns(2)
                                with col_p1:
                                    # Actual vs Predicted Scatter
                                    fig_avp = px.scatter(plot_df, x='Actual', y='Predicted', title="Actual vs. Predicted Prices (Test Set)",
                                                        opacity=0.7, hover_data={'Residuals': ':.2f'},
                                                        trendline="ols", trendline_color_override=NEON_PINK)
                                    fig_avp.add_shape(type="line", x0=plot_df['Actual'].min(), y0=plot_df['Actual'].min(),
                                                    x1=plot_df['Actual'].max(), y1=plot_df['Actual'].max(),
                                                    line=dict(color="grey", width=1, dash="dash")) # y=x line
                                    fig_avp.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,17,17,0.9)')
                                    st.plotly_chart(fig_avp, use_container_width=True)
                                with col_p2:
                                    # Residual Plot
                                    fig_res = px.scatter(plot_df, x='Predicted', y='Residuals', title="Residuals vs. Predicted Prices (Test Set)",
                                                        opacity=0.7)
                                    fig_res.add_hline(y=0, line=dict(color=NEON_PINK, width=1, dash="dash")) # Zero residual line
                                    fig_res.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,17,17,0.9)')
                                    st.plotly_chart(fig_res, use_container_width=True)
                            except Exception as plot_err:
                                st.error(f"Error displaying regression performance plots: {plot_err}")
                                st.error(traceback.format_exc())
                         else:
                             st.info("Regression performance plot data not available.")


                elif "Logistic Regression" in st.session_state.selected_model:
                    st.markdown(f"""
                    <div class='dashboard-section gradient-border'>
                        <h3 style='color:{NEON_PINK}; text-align:center;'>Trend Prediction Results (Logistic Regression)</h3>

                    """, unsafe_allow_html=True)
                    # --- Logistic Regression Description ---
                    st.markdown(f"""
                    <div class='hover-card' style='background-color:rgba(0,0,0,0.7); border-left:4px solid {NEON_GREEN}; margin-bottom:10px;'>
                        <b>What is Logistic Regression?</b><br>
                        Logistic regression is a statistical method used for binary classification. It estimates the probability that a target variable belongs to a particular class. In this app, it predicts the <b>probability that the stock price will go UP or DOWN </b> based on historical data and technical indicators.
                    </div>
                    """, unsafe_allow_html=True)
                    try:
                        display_trend_prediction(st.session_state.ml_results) # Includes performance plots in expander
                    except Exception as plot_err:
                        st.error(f"Error displaying trend prediction results: {plot_err}")
                        st.error(traceback.format_exc())


                # --- Plot Customization and Insights ---
                st.markdown("---")
                with st.expander("⚙️ Model Insights & Plot Customization", expanded=True):
                     # Colorscale Selection
                     st.markdown("<h4 style='color:white;'>Customize Plot Colors</h4>", unsafe_allow_html=True)
                     col_cs1, col_cs2 = st.columns(2)
                     with col_cs1:
                          # Filter for diverging colorscales
                         diverging_colorscales = [cs for cs in px.colors.named_colorscales() if cs.lower() in ['rdbu', 'plotly3', 'prgn', 'piyg', 'picnic', 'portland', 'puor', 'rdgy', 'rdylbu', 'rdylgn', 'spectral', 'seismic', 'brbg', 'balance', 'curl', 'delta', 'geyser']]
                         diverging_colorscales = [cs.lower() for cs in diverging_colorscales] # Ensure all are lowercase
                         default_div_cs = st.session_state.selected_diverging_colorscale.lower() if st.session_state.selected_diverging_colorscale else 'rdbu'
                         if default_div_cs not in diverging_colorscales: default_div_cs = 'rdbu'
                         st.session_state.selected_diverging_colorscale = st.selectbox(
                              "Heatmap Colorscale (Correlation)",
                              diverging_colorscales,
                              index=diverging_colorscales.index(default_div_cs),
                              key="select_diverging_cs"
                         )
                     with col_cs2:
                          # Filter for scales suitable for coefficients (can be diverging or sequential)
                         coeff_colorscales = [cs.lower() for cs in px.colors.named_colorscales()]
                         default_coeff_cs = st.session_state.selected_coefficient_colorscale.lower() if st.session_state.selected_coefficient_colorscale else 'picnic'
                         if default_coeff_cs not in coeff_colorscales:
                             default_coeff_cs = 'picnic'
                         st.session_state.selected_coefficient_colorscale = st.selectbox(
                             "Feature Influence Colorscale (Coefficients)",
                             coeff_colorscales,
                             index=coeff_colorscales.index(default_coeff_cs),
                             key="select_coeff_cs"
                         )

                     st.markdown("<hr style='border-top: 1px solid rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)

                     # --- Feature Importance / Coefficients ---
                     st.markdown(f"""
                     <div >
                         <h4 style='color:{NEON_PINK};'>Model Insights: Feature Influence</h4>
                         <p style='font-size:0.9em;'>Factors influencing the selected model's predictions. Coefficients indicate the relationship's direction and magnitude (on scaled data).</p>
                     </div>
                     """, unsafe_allow_html=True)

                     model_to_explain = None
                     # Make sure corr_df exists and has columns before trying to access them
                     features_list = []
                     if st.session_state.corr_df is not None and not st.session_state.corr_df.empty:
                         try:
                            features_list = st.session_state.corr_df.drop(columns=['Price_5d_Ahead', 'Trend_5d_Ahead'], errors='ignore').columns.tolist()
                         except KeyError:
                             st.warning("Could not find target columns in correlation dataframe for feature list generation.")
                             # Fallback: use all columns except the last two assumed targets
                             if len(st.session_state.corr_df.columns) > 2:
                                 features_list = list(st.session_state.corr_df.columns[:-2])
                             else:
                                 features_list = [] # Cannot determine features

                     current_trained_models = st.session_state.trained_models
                     coeff_model_name = "" # To store which model's coefficients are shown

                     # Check if features_list is populated before proceeding
                     if not features_list:
                          st.warning("Could not determine feature list for importance plot.")
                     else:
                         # Display for Linear Regression
                         if current_trained_models and 'regression' in current_trained_models and current_trained_models['regression'] is not None:
                             model_to_explain = current_trained_models['regression']
                             coeff_model_name = "Linear Regression"
                             try:
                                  coeffs = model_to_explain.coef_
                                  if len(coeffs) == len(features_list):
                                      importance_df = pd.DataFrame({'Feature': features_list, 'Coefficient': coeffs})
                                      importance_df['Abs_Coefficient'] = np.abs(importance_df['Coefficient'])
                                      # Sort and select top N features
                                      importance_df = importance_df.nlargest(15, 'Abs_Coefficient')

                                      fig_imp = px.bar(importance_df.sort_values(by='Coefficient'), x='Coefficient', y='Feature', orientation='h',
                                                    title=f"Top 15 Feature Coefficients ({coeff_model_name})",
                                                    color='Coefficient',
                                                    color_continuous_scale=st.session_state.selected_coefficient_colorscale) # Use selected colorscale
                                      fig_imp.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,17,17,0.9)', yaxis={'categoryorder':'total ascending'})
                                      st.plotly_chart(fig_imp, use_container_width=True)
                                  else:
                                      st.warning(f"Mismatch between number of coefficients ({len(coeffs)}) and features ({len(features_list)}) for Linear Regression.")

                             except Exception as e:
                                  st.warning(f"Could not display feature coefficients for Linear Regression: {e}")
                                  st.error(traceback.format_exc())

                         # Display for Logistic Regression
                         if current_trained_models and 'classification' in current_trained_models and current_trained_models['classification'] is not None:
                             model_to_explain = current_trained_models['classification']
                             coeff_model_name = "Logistic Regression"
                             try:
                                 # Logistic Regression coefficients relate to log-odds
                                 coeffs = model_to_explain.coef_[0] # Coeffs are usually in coef_[0]
                                 if len(coeffs) == len(features_list):
                                     importance_df = pd.DataFrame({'Feature': features_list, 'Coefficient (Log-Odds)': coeffs})
                                     importance_df['Abs_Coefficient'] = np.abs(importance_df['Coefficient (Log-Odds)'])
                                     # Sort and select top N features
                                     importance_df = importance_df.nlargest(15, 'Abs_Coefficient')


                                     fig_imp = px.bar(importance_df.sort_values(by='Coefficient (Log-Odds)'), x='Coefficient (Log-Odds)', y='Feature', orientation='h',
                                                     title=f"Top 15 Feature Coefficients ({coeff_model_name})",
                                                     color='Coefficient (Log-Odds)',
                                                     color_continuous_scale=st.session_state.selected_coefficient_colorscale) # Use selected colorscale
                                     fig_imp.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,17,17,0.9)', yaxis={'categoryorder':'total ascending'})
                                     st.plotly_chart(fig_imp, use_container_width=True)
                                 else:
                                     st.warning(f"Mismatch between number of coefficients ({len(coeffs)}) and features ({len(features_list)}) for Logistic Regression.")

                             except Exception as e:
                                 st.warning(f"Could not display feature coefficients for Logistic Regression: {e}")
                                 st.error(traceback.format_exc())

                         # Message if no models available
                         if not (current_trained_models and ('regression' in current_trained_models or 'classification' in current_trained_models)):
                              st.info("Feature influence chart unavailable. Run the ML pipeline first.")


                     # --- Correlation Heatmap ---
                     st.markdown("<hr style='border-top: 1px solid rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)
                     st.markdown(f"""
                     <div>
                         <h4 style='color:{NEON_PINK};'>Feature Correlation Analysis</h4>
                         <p style='font-size:0.9em;'>Heatmap showing linear correlations between features and the target variables (Price & Trend 5 days ahead). Use the colorscale selector above to change the theme.</p>
                     </div>
                     """, unsafe_allow_html=True)

                     corr_df_display = st.session_state.corr_df # Use the stored correlation dataframe
                     if corr_df_display is not None and not corr_df_display.empty:
                         try:
                             # Ensure only numeric columns are used for correlation
                             numeric_corr_df = corr_df_display.select_dtypes(include=np.number)
                             if numeric_corr_df.empty:
                                 st.warning("No numeric columns found for correlation calculation.")
                             else:
                                corr_matrix = numeric_corr_df.corr()
                                # Select only correlations with the target variables for clarity
                                target_cols = ['Price_5d_Ahead', 'Trend_5d_Ahead']
                                available_target_cols = [col for col in target_cols if col in corr_matrix.columns]

                                if not available_target_cols:
                                     st.warning("Target columns ('Price_5d_Ahead', 'Trend_5d_Ahead') not found in correlation matrix.")
                                else:
                                    # Drop the target columns from the index (rows) to avoid self-correlation
                                    target_corr = corr_matrix[available_target_cols].drop(available_target_cols, errors='ignore')

                                    if not target_corr.empty:
                                        fig_corr = go.Figure(data=go.Heatmap(
                                            z=target_corr.values,
                                            x=target_corr.columns,
                                            y=target_corr.index,
                                            colorscale=st.session_state.selected_diverging_colorscale, # Use selected colorscale
                                            zmin=-1, zmax=1, # Fix scale from -1 to 1
                                            colorbar=dict(title='Corr', thickness=10, len=0.8) # Slimmer colorbar
                                        ))
                                        fig_corr.update_layout(
                                            title='Correlation of Features with 5-Day Ahead Targets',
                                            template="plotly_dark",
                                            height=max(400, len(target_corr.index) * 20), # Adjust height factor
                                            yaxis_nticks=len(target_corr.index),
                                            xaxis_tickangle=-30,
                                            margin=dict(l=150, r=20, t=50, b=100), # Adjust margins
                                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,17,17,0.9)'
                                        )
                                        st.plotly_chart(fig_corr, use_container_width=True)
                                    else:
                                        st.warning("Could not calculate target correlations (result was empty after processing).")


                         except Exception as e:
                             st.warning(f"Could not display correlation heatmap: {e}")
                             st.error(traceback.format_exc())
                     else:
                         st.info("Correlation data not available (ML Pipeline might need to be run).")


            elif st.session_state.ml_error:
                 # Display the ML error prominently if pipeline failed
                 st.error(f"Machine Learning Pipeline Execution Failed:")
                 st.error(st.session_state.ml_error)
                 st.info("This might be due to insufficient data for the selected period, data quality issues (check 'View Raw Data' in Tab 1), or internal errors. Try adjusting the time period, checking the input data, or re-running the pipeline.")
            else:
                 # Case where ML hasn't run yet
                 st.info("Machine learning analysis has not been run yet for the current data. Click 'Re-run ML Pipeline' above to generate predictions.")


# --- Entry Point ---
if __name__ == "__main__":
    main()

# --- END OF FILE app.py ---