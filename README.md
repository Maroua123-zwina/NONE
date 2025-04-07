import pandas as pd
import numpy as np
import io
from scipy.stats import gamma, norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- 1. Data Loading and Preprocessing ---

# Function to load simple CSVs
def load_simple_csv(filename, date_col='Date', value_col=None):
    """Loads CSV with Date and Value columns."""
    try:
        df = pd.read_csv(filename)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        if value_col:
            df = df[[value_col]] # Keep only the specified value column
        df = df.resample('MS').mean() # Ensure monthly start frequency
        print(f"Loaded {filename}: {df.shape[0]} rows")
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Function to load and process the POWER data
def load_power_data(filename):
    """Loads and reshapes the NASA POWER data."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Find header end
        header_end_index = -1
        for i, line in enumerate(lines):
            if "-END HEADER-" in line:
                header_end_index = i + 1
                break

        if header_end_index == -1:
            raise ValueError("Could not find '-END HEADER-' in POWER file")

        # Read data into pandas DataFrame, skipping header
        data_str = "".join(lines[header_end_index:])
        df_power = pd.read_csv(io.StringIO(data_str), delimiter=';')

        # Clean column names (remove potential extra ; characters)
        df_power.columns = [col.strip() for col in df_power.columns[:15]] + ['DROP_ME'] * (len(df_power.columns) - 15)
        df_power = df_power.drop(columns=[col for col in df_power.columns if 'DROP_ME' in col or col == 'ANN'])

        # Rename columns for clarity
        df_power = df_power.rename(columns={'PARAMETER': 'Parameter', 'YEAR': 'Year'})

        # Melt the DataFrame
        df_melted = df_power.melt(id_vars=['Parameter', 'Year'],
                                  var_name='Month',
                                  value_name='Value')

        # Map month abbreviations to numbers
        month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                     'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        df_melted['MonthNum'] = df_melted['Month'].map(month_map)

        # Create DateTimeIndex
        # Handle potential errors during date creation
        df_melted = df_melted.dropna(subset=['Year', 'MonthNum']) # Drop rows where month mapping failed
        df_melted['Year'] = df_melted['Year'].astype(int)
        df_melted['Date'] = pd.to_datetime(df_melted[['Year', 'MonthNum']].assign(DAY=1).rename(columns={'MonthNum': 'MONTH'}))

        # Pivot the table
        df_pivot = df_melted.pivot_table(index='Date', columns='Parameter', values='Value')

        # Rename columns for consistency
        df_pivot = df_pivot.rename(columns={
            'PRECTOTCORR': 'Precipitation',
            'T2M': 'Temperature'
        })

        # Convert to numeric, coercing errors (like -999) to NaN
        for col in df_pivot.columns:
            df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce')

        # Replace potential placeholders like -999.0 with NaN
        df_pivot = df_pivot.replace(-999.0, np.nan)

        print(f"Loaded and processed {filename}: {df_pivot.shape[0]} rows")
        return df_pivot

    except Exception as e:
        print(f"Error loading or processing {filename}: {e}")
        return None

# Load data
ndvi_file = 'Marrakesh_NDVI_2010_2024.csv'
sm_file = 'Marrakesh_Soil_Moisture_2010_2025.csv'
# Use one of the identical POWER files
power_file = 'POWER_Point_Monthly_20100101_20251231_031d64N_008d00W_LST.csv'

df_ndvi = load_simple_csv(ndvi_file, value_col='NDVI')
df_sm = load_simple_csv(sm_file, value_col='Soil Moisture (m³/m³)')
df_power = load_power_data(power_file)

# Check if data loaded successfully
if df_ndvi is None or df_sm is None or df_power is None:
    raise SystemExit("Error loading one or more data files. Exiting.")

# Merge DataFrames
df_merged = pd.concat([df_ndvi, df_sm, df_power], axis=1)
df_merged = df_merged.rename(columns={'Soil Moisture (m³/m³)': 'Soil_Moisture'})

# Define the date range based on the user request (2010-2023 for training/evaluation)
# We need data slightly before 2010 for SPI calculation lags
start_date_spi_calc = '2009-01-01' # Adjust if using longer SPI timescale
end_date_analysis = '2023-12-01'
start_date_analysis = '2010-01-01'


df_merged = df_merged[start_date_spi_calc:] # Keep data from before 2010 if needed for SPI lag

print("\nMerged DataFrame head:")
print(df_merged.head())
print("\nMerged DataFrame tail:")
print(df_merged.tail())
print("\nData Info:")
df_merged.info()
print("\nMissing values before interpolation:")
print(df_merged.isnull().sum())

# --- Interpolate missing values (e.g., from -999 in POWER data or gaps) ---
# Using linear interpolation for simplicity, consider more advanced methods if needed
df_merged = df_merged.interpolate(method='linear', limit_direction='both')

print("\nMissing values after interpolation:")
print(df_merged.isnull().sum())

# Drop rows with any remaining NaNs (should ideally be none after interpolation)
df_merged.dropna(inplace=True)

# --- 2. Calculate SPI ---

def calculate_spi(series, timescale, fit_period_start=None, fit_period_end=None):
    """Calculates SPI using a Gamma distribution fit."""
    # Calculate rolling sum for the timescale
    rolling_prec = series.rolling(window=timescale, min_periods=timescale).sum()

    spi_values = pd.Series(index=series.index, dtype=float)

    # Determine the period for fitting the distribution
    fit_series = series.copy()
    if fit_period_start:
        fit_series = fit_series[fit_series.index >= pd.to_datetime(fit_period_start)]
    if fit_period_end:
        fit_series = fit_series[fit_series.index <= pd.to_datetime(fit_period_end)]

    fit_rolling_prec = fit_series.rolling(window=timescale, min_periods=timescale).sum().dropna()


    for month in range(1, 13):
        # Filter data for the specific month for fitting
        monthly_data_fit = fit_rolling_prec[fit_rolling_prec.index.month == month]

        # Filter data for the specific month for transformation (entire series)
        monthly_data_transform = rolling_prec[rolling_prec.index.month == month]

        if monthly_data_fit.empty or monthly_data_fit.nunique() < 2: # Need data to fit
             print(f"Warning: Insufficient data for month {month} to fit Gamma dist. SPI will be NaN.")
             spi_values[monthly_data_transform.index] = np.nan
             continue

        # Remove zeros for fitting Gamma shape/scale, but keep track for probability calculation
        monthly_data_fit_no_zeros = monthly_data_fit[monthly_data_fit > 0]
        prob_zero = (monthly_data_fit == 0).mean()

        if monthly_data_fit_no_zeros.empty: # All zeros in fit period for this month
            print(f"Warning: Only zero precipitation for month {month} in fit period. SPI will be NaN for non-zeros, 0 for zeros.")
            spi_values[monthly_data_transform[monthly_data_transform == 0].index] = norm.ppf(0.5) # Assign neutral SPI for zeros
            spi_values[monthly_data_transform[monthly_data_transform > 0].index] = np.nan # Can't calculate for non-zeros
            continue

        try:
            # Fit Gamma distribution
            params = gamma.fit(monthly_data_fit_no_zeros, floc=0) # floc=0 assumes lower bound is 0
            shape, loc, scale = params # loc should be near 0

            # Calculate CDF for the transformation data
            cdf_non_zero = gamma.cdf(monthly_data_transform[monthly_data_transform > 0], *params)
            cdf = prob_zero + (1 - prob_zero) * cdf_non_zero

            # Handle potential CDF edge cases (0 or 1) before norm.ppf
            epsilon = 1e-6 # Small value to avoid infinity
            cdf = np.clip(cdf, epsilon, 1 - epsilon)

            # Transform CDF to SPI (Z-score)
            spi_month = norm.ppf(cdf)

            # Assign SPI values
            spi_values[monthly_data_transform[monthly_data_transform > 0].index] = spi_month

            # Handle original zeros in the transformation data
            if prob_zero > 0:
                 # Assign the SPI value corresponding to the midpoint of the zero probability portion
                 spi_values[monthly_data_transform[monthly_data_transform == 0].index] = norm.ppf(prob_zero / 2)
            else:
                 # If no zeros in fit period, but zero occurs in transform period, assign low SPI
                 spi_values[monthly_data_transform[monthly_data_transform == 0].index] = norm.ppf(epsilon) # Very dry


        except Exception as e:
            print(f"Error fitting Gamma or calculating SPI for month {month}: {e}")
            spi_values[monthly_data_transform.index] = np.nan # Assign NaN on error

    return spi_values

# Calculate SPI-3 (using data up to the end_date_analysis for fitting)
spi_timescale = 3
df_merged[f'SPI_{spi_timescale}'] = calculate_spi(df_merged['Precipitation'],
                                                timescale=spi_timescale,
                                                fit_period_end=end_date_analysis) # Fit on historical period

# Drop initial NaNs resulting from rolling calculations and keep analysis period
df_analysis = df_merged[start_date_analysis:end_date_analysis].copy()
df_analysis.dropna(inplace=True) # Drop any NaNs from SPI calculation issues

print(f"\nDataFrame for analysis ({start_date_analysis} to {end_date_analysis}) after SPI calculation:")
print(df_analysis.head())
print(df_analysis.tail())
print(f"\nRemaining NaNs in analysis data: {df_analysis.isnull().sum().sum()}")

# Target variable
target_col = f'SPI_{spi_timescale}'

# --- 3. Feature Engineering ---

# Create lagged features
n_lags = 6 # Number of past months to consider for prediction
for i in range(1, n_lags + 1):
    df_analysis[f'SPI_lag_{i}'] = df_analysis[target_col].shift(i)
    df_analysis[f'Precip_lag_{i}'] = df_analysis['Precipitation'].shift(i)
    df_analysis[f'NDVI_lag_{i}'] = df_analysis['NDVI'].shift(i)
    df_analysis[f'SM_lag_{i}'] = df_analysis['Soil_Moisture'].shift(i)
    # df_analysis[f'Temp_lag_{i}'] = df_analysis['Temperature'].shift(i) # Optional: add lagged temperature

# Add cyclical time features (optional but can help)
# df_analysis['month_sin'] = np.sin(2 * np.pi * df_analysis.index.month / 12)
# df_analysis['month_cos'] = np.cos(2 * np.pi * df_analysis.index.month / 12)

# Drop rows with NaNs created by lagging
df_analysis.dropna(inplace=True)

# Select features for LSTM
# Include lagged SPI, lagged Precip, lagged NDVI, lagged SM. Exclude current Precip/NDVI/SM
# as they wouldn't be known when predicting the current month's SPI.
features = [col for i in range(1, n_lags + 1) for col in df_analysis.columns if col.endswith(f'lag_{i}')]
# features += ['month_sin', 'month_cos'] # Add time features if used
print(f"\nFeatures used for LSTM ({len(features)}): {features}")

# --- Data Scaling ---
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Scale features
df_analysis[features] = scaler_features.fit_transform(df_analysis[features])

# Scale target separately
df_analysis[target_col] = scaler_target.fit_transform(df_analysis[[target_col]])

print("\nScaled data sample:")
print(df_analysis[[target_col] + features].head())


# --- 4. Hybrid Model Implementation ---

# --- Train/Test Split ---
test_size = 0.20 # Use last 20% for testing
split_index = int(len(df_analysis) * (1 - test_size))
df_train = df_analysis.iloc[:split_index]
df_test = df_analysis.iloc[split_index:]

print(f"\nTraining data shape: {df_train.shape}")
print(f"Testing data shape: {df_test.shape}")

X_train = df_train[features].values
y_train = df_train[target_col].values
X_test = df_test[features].values
y_test = df_test[target_col].values


# --- LSTM Model ---
# Reshape data for LSTM [samples, timesteps, features]
# Here, we treat each month's lagged features as a single timestep input
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM model
lstm_model = Sequential([
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])), # Explicit Input layer
    LSTM(units=50, activation='relu', return_sequences=True), # Keep return_sequences=True if stacking LSTM layers
    LSTM(units=50, activation='relu'),
    Dense(units=1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

# Train LSTM model
history = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=100,         # Adjust epochs as needed
    batch_size=32,      # Adjust batch size
    validation_split=0.1, # Use part of training data for validation during training
    verbose=1,          # Set to 0 to reduce output during training
    shuffle=False       # Important for time series
)

# --- LSTM Predictions and Residuals ---
lstm_pred_scaled_train = lstm_model.predict(X_train_lstm)
lstm_pred_scaled_test = lstm_model.predict(X_test_lstm)

# Inverse scale LSTM predictions
lstm_pred_train = scaler_target.inverse_transform(lstm_pred_scaled_train)
lstm_pred_test = scaler_target.inverse_transform(lstm_pred_scaled_test)

# Inverse scale actual values to calculate residuals in original scale
y_train_orig = scaler_target.inverse_transform(y_train.reshape(-1, 1))
y_test_orig = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate residuals
train_residuals = y_train_orig.flatten() - lstm_pred_train.flatten()
test_residuals = y_test_orig.flatten() - lstm_pred_test.flatten()


# --- ARIMA Model on Residuals ---
# Use auto_arima to find best ARIMA order for the training residuals
try:
    auto_arima_model = pm.auto_arima(train_residuals,
                                     start_p=1, start_q=1,
                                     max_p=5, max_q=5, m=12, # Consider seasonality (m=12 for monthly)
                                     start_P=0, seasonal=True, # Enable seasonal component search
                                     d=0,           # Residuals should be stationary, so d=0 usually
                                     trace=True, error_action='ignore',
                                     suppress_warnings=True, stepwise=True)

    print("\nAuto ARIMA Best Model Summary:")
    print(auto_arima_model.summary())
    arima_order = auto_arima_model.order
    seasonal_order = auto_arima_model.seasonal_order

    # Fit the best ARIMA model found by auto_arima on training residuals
    arima_model = ARIMA(train_residuals, order=arima_order, seasonal_order=seasonal_order)
    arima_fit = arima_model.fit()
    print("\nARIMA Model Fit Summary:")
    print(arima_fit.summary())


    # Forecast residuals on the test set period
    # Use dynamic=False to use true lagged residuals within the training period for fitting consistency,
    # then forecast n_periods ahead matching the test set size.
    arima_pred_test = arima_fit.predict(start=len(train_residuals), end=len(train_residuals) + len(test_residuals) - 1)
    # Ensure the prediction length matches the test set
    if len(arima_pred_test) != len(test_residuals):
         print(f"Warning: ARIMA prediction length ({len(arima_pred_test)}) mismatch with test residuals ({len(test_residuals)}). Adjusting forecast.")
         # Adjust using forecast if predict didn't work as expected for future steps
         arima_pred_test = arima_fit.forecast(steps=len(test_residuals))


except Exception as e:
    print(f"\nError during auto_arima or ARIMA fitting: {e}")
    print("Proceeding without ARIMA correction.")
    arima_pred_test = np.zeros_like(test_residuals) # No correction if ARIMA fails


# --- Combine Predictions ---
hybrid_pred_test = lstm_pred_test.flatten() + arima_pred_test.flatten()

# Create DataFrame for results
results_df = pd.DataFrame({
    'Actual_SPI': y_test_orig.flatten(),
    'LSTM_Pred': lstm_pred_test.flatten(),
    'ARIMA_Residual_Pred': arima_pred_test.flatten(),
    'Hybrid_Pred': hybrid_pred_test
}, index=df_test.index)


# --- 5. Evaluation ---
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}

print("\n--- Evaluation on Test Set ---")
lstm_metrics = calculate_metrics(results_df['Actual_SPI'], results_df['LSTM_Pred'])
hybrid_metrics = calculate_metrics(results_df['Actual_SPI'], results_df['Hybrid_Pred'])

print(f"LSTM Model: RMSE={lstm_metrics['RMSE']:.4f}, MAE={lstm_metrics['MAE']:.4f}")
print(f"Hybrid (LSTM+ARIMA) Model: RMSE={hybrid_metrics['RMSE']:.4f}, MAE={hybrid_metrics['MAE']:.4f}")

# --- 6. Forecasting Future SPI ---
# NOTE: Forecasting beyond the available data (Dec 2023 in this analysis setup)
# requires forecasting the input features (NDVI, Soil Moisture, Precipitation lags)
# first, which adds complexity. Here, we'll just show the structure for predicting
# the next step based on the last available data point.

# Get the last actual data point's features (from the original unscaled df_analysis)
last_data_point_orig = df_merged.loc[df_analysis.index[-1]].copy() # Get original data before scaling/lagging

# Create future features based on the last point (simplistic example)
# Need to shift existing lags and predict/estimate the most recent lag (lag_1)
future_features_dict = {}

# Shift existing lags
for i in range(n_lags, 1, -1):
    future_features_dict[f'SPI_lag_{i}'] = last_data_point_orig.get(f'SPI_lag_{i-1}', 0) # Use previous lag
    future_features_dict[f'Precip_lag_{i}'] = last_data_point_orig.get(f'Precip_lag_{i-1}', 0)
    future_features_dict[f'NDVI_lag_{i}'] = last_data_point_orig.get(f'NDVI_lag_{i-1}', 0)
    future_features_dict[f'SM_lag_{i}'] = last_data_point_orig.get(f'SM_lag_{i-1}', 0)

# Use the actual last values for lag_1
future_features_dict['SPI_lag_1'] = last_data_point_orig[target_col]
future_features_dict['Precip_lag_1'] = last_data_point_orig['Precipitation']
future_features_dict['NDVI_lag_1'] = last_data_point_orig['NDVI']
future_features_dict['SM_lag_1'] = last_data_point_orig['Soil_Moisture']

# Convert to DataFrame and ensure correct order
future_features_df = pd.DataFrame([future_features_dict])[features] # Select only feature columns in correct order

# Scale future features using the fitted scaler
future_features_scaled = scaler_features.transform(future_features_df)

# Reshape for LSTM
future_features_lstm = future_features_scaled.reshape((1, 1, len(features)))

# Predict next step LSTM
lstm_pred_next_scaled = lstm_model.predict(future_features_lstm)
lstm_pred_next = scaler_target.inverse_transform(lstm_pred_next_scaled)

# Predict next step ARIMA residual (using the fitted ARIMA model)
try:
    arima_pred_next = arima_fit.forecast(steps=1)[0]
except Exception as e:
    print(f"Could not forecast next ARIMA step: {e}")
    arima_pred_next = 0 # Assume no correction if forecast fails

# Combine for hybrid prediction
hybrid_pred_next = lstm_pred_next.flatten()[0] + arima_pred_next

forecast_date = df_analysis.index[-1] + pd.DateOffset(months=1)
print(f"\n--- Forecast for {forecast_date.strftime('%Y-%m-%d')} ---")
print(f"Predicted SPI (Hybrid): {hybrid_pred_next:.4f}")


# --- 7. Drought Interpretation ---
def interpret_spi(spi_value):
    if spi_value >= 2.0:
        return "Extremely Wet"
    elif 1.5 <= spi_value < 2.0:
        return "Severely Wet"
    elif 1.0 <= spi_value < 1.5:
        return "Moderately Wet"
    elif 0 <= spi_value < 1.0:
        return "Near Normal (Slightly Wet)"
    elif -0.99 <= spi_value < 0:
         return "Near Normal (Slightly Dry)"
    elif -1.49 <= spi_value < -0.99:
        return "Moderate Drought"
    elif -1.99 <= spi_value < -1.49:
        return "Severe Drought"
    elif spi_value < -1.99:
        return "Extreme Drought"
    else:
        return "Undefined"

print(f"Interpretation: {interpret_spi(hybrid_pred_next)}")

# Add interpretation to results dataframe
results_df['Drought_Category_Actual'] = results_df['Actual_SPI'].apply(interpret_spi)
results_df['Drought_Category_Predicted'] = results_df['Hybrid_Pred'].apply(interpret_spi)

print("\nTest Set Results with Drought Interpretation:")
print(results_df[['Actual_SPI', 'Hybrid_Pred', 'Drought_Category_Actual', 'Drought_Category_Predicted']].head())
print(results_df[['Actual_SPI', 'Hybrid_Pred', 'Drought_Category_Actual', 'Drought_Category_Predicted']].tail())


# --- Visualization ---
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1) # Plot training performance (optional)
plt.plot(df_train.index, y_train_orig, label='Actual SPI (Train)', color='blue', alpha=0.7)
plt.plot(df_train.index, lstm_pred_train + arima_fit.fittedvalues, label='Hybrid Pred (Train)', color='orange', linestyle='--')
plt.title('SPI Prediction - Training Set')
plt.ylabel('SPI')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2) # Plot test performance
plt.plot(results_df.index, results_df['Actual_SPI'], label='Actual SPI (Test)', color='blue')
#plt.plot(results_df.index, results_df['LSTM_Pred'], label='LSTM Prediction', color='green', linestyle=':', alpha=0.7)
plt.plot(results_df.index, results_df['Hybrid_Pred'], label='Hybrid Prediction (LSTM+ARIMA)', color='red', linestyle='--')
plt.title('SPI Prediction - Test Set')
plt.xlabel('Date')
plt.ylabel('SPI')
plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
plt.axhline(-1, color='orange', linestyle='--', linewidth=0.8, label='Moderate Drought Threshold')
plt.axhline(-1.5, color='red', linestyle='--', linewidth=0.8, label='Severe Drought Threshold')
plt.axhline(-2, color='darkred', linestyle='--', linewidth=0.8, label='Extreme Drought Threshold')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot Residuals (optional)
plt.figure(figsize=(15, 5))
plt.plot(df_test.index, test_residuals, label='LSTM Test Residuals', alpha=0.8)
plt.plot(df_test.index, arima_pred_test, label='ARIMA Prediction of Residuals', linestyle='--')
plt.title('LSTM Test Residuals and ARIMA Fit')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.legend()
plt.grid(True)
plt.show()
