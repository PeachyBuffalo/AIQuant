from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import sys

print(f"Python executable: {sys.executable}")
print(f"yfinance version: {yf.__version__}")

warnings.filterwarnings('ignore')

# Helper function for RSI calculation
def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def detect_trend(data, window=20):
    """Detect trend direction and strength"""
    
    # Adjust window if dataset is too small
    if len(data) < window:
        window = max(1, len(data) // 2)
    
    # Price trend
    if len(data) > window:
        price_trend = (data['Close'] - data['Close'].shift(window)) / data['Close'].shift(window)
    else:
        price_trend = pd.Series([np.nan] * len(data), index=data.index)
    
    # Moving average trend
    if len(data) >= 10:
        sma_short = data['Close'].rolling(window=min(10, len(data))).mean()
        sma_long = data['Close'].rolling(window=window).mean()
        ma_trend = (sma_short - sma_long) / sma_long
    else:
        ma_trend = pd.Series([np.nan] * len(data), index=data.index)
    
    # Volume trend
    if len(data) > window:
        volume_trend = (data['Volume'] - data['Volume'].rolling(window=window).mean()) / data['Volume'].rolling(window=window).mean()
    else:
        volume_trend = pd.Series([np.nan] * len(data), index=data.index)
    
    # Trend classification
    if len(data) > window:
        trend_direction = np.where(price_trend > 0.02, 'Uptrend', 
                                  np.where(price_trend < -0.02, 'Downtrend', 'Sideways'))
        trend_strength = abs(price_trend) * 100
    else:
        trend_direction = pd.Series(['Sideways'] * len(data), index=data.index)
        trend_strength = pd.Series([0.0] * len(data), index=data.index)
    
    # Ensure all Series have the same index
    price_trend = pd.Series(price_trend.values.flatten(), index=data.index)
    ma_trend = pd.Series(ma_trend.values.flatten(), index=data.index)
    volume_trend = pd.Series(volume_trend.values.flatten(), index=data.index)
    if isinstance(trend_direction, np.ndarray):
        trend_direction = pd.Series(trend_direction.flatten(), index=data.index)
    else:
        trend_direction = pd.Series(trend_direction, index=data.index)
    trend_strength = pd.Series(trend_strength.values.flatten(), index=data.index)
    
    return pd.DataFrame({
        'Price_Trend': price_trend,
        'MA_Trend': ma_trend,
        'Volume_Trend': volume_trend,
        'Trend_Direction': trend_direction,
        'Trend_Strength': trend_strength
    })

def analyze_support_resistance(data, window=20):
    """Identify support and resistance levels"""
    # Check if required columns exist
    required_columns = ['High', 'Low']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        # Return empty DataFrame with same index
        return pd.DataFrame({
            'Resistance': [False] * len(data),
            'Support': [False] * len(data),
            'Resistance_Level': [np.nan] * len(data),
            'Support_Level': [np.nan] * len(data)
        }, index=data.index)
    
    highs = data['High'].rolling(window=window, center=True).max()
    lows = data['Low'].rolling(window=window, center=True).min()
    
    # Resistance levels (local highs)
    resistance = data['High'] == highs
    # Support levels (local lows)
    support = data['Low'] == lows
    
    return pd.DataFrame({
        'Resistance': resistance,
        'Support': support,
        'Resistance_Level': np.where(resistance, data['High'], np.nan),
        'Support_Level': np.where(support, data['Low'], np.nan)
    })

def calculate_atr(data, window=14):
    """Calculate Average True Range"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(window=window).mean()

def calculate_stochastic(data, window=14):
    """Calculate Stochastic Oscillator %K"""
    lowest_low = data['Low'].rolling(window=window).min()
    highest_high = data['High'].rolling(window=window).max()
    return ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100

def calculate_obv(data):
    """Calculate On-Balance Volume"""
    if len(data) < 2:
        return pd.Series([np.nan]*len(data), index=data.index)
    obv = pd.Series(index=data.index, dtype=float)
    obv.iloc[0] = data['Volume'].iloc[0]
    for i in range(1, len(data)):
        prev_close = data['Close'].iloc[i-1].item()
        curr_close = data['Close'].iloc[i].item()
        if curr_close > prev_close:
            obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i].item()
        elif curr_close < prev_close:
            obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i].item()
        else:
            obv.iloc[i] = obv.iloc[i-1]
    return obv

# ============================================================================
# TREND DETECTION FOR STOCKS, CRYPTO, AND OPTIONS
# ============================================================================

print("=== AI-Driven Trend Detection System ===")
print("Analyzing Trends for Stocks, Cryptocurrencies, and Options\n")

# Define assets to analyze
assets = {
    'stocks': ['TSLA', 'AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META', 'AMZN', 'NFLX']
}
# Load data for trend analysis
print("1. Loading Market Data for Trend Analysis...")
data_dict = {}
start_date = '2023-01-01'
end_date = '2024-01-01'

# Track if we got any real data
real_data_loaded = False

for category, symbols in assets.items():
    print(f"\n   Loading {category.upper()} data...")
    for symbol in symbols:
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                data_dict[symbol] = data
                real_data_loaded = True
                print(f"   ✓ {symbol}: {len(data)} records")
            else:
                print(f"   ✗ {symbol}: No data available")
        except Exception as e:
            print(f"   ✗ {symbol}: Error - {str(e)}")

# If no real data was loaded, create synthetic data for testing
if not real_data_loaded:
    print("\n   ⚠️  No real data available. Creating synthetic data for testing...")
    dates = pd.date_range(start_date, end_date, freq='D')
    
    for category, symbols in assets.items():
        for symbol in symbols:
            # Generate realistic synthetic stock data
            np.random.seed(hash(symbol) % 1000)
            
            # Start with a base price and add realistic price movements
            base_price = np.random.uniform(50, 300)
            prices = [base_price]
            
            for i in range(1, len(dates)):
                # Add daily returns with some trend and volatility
                daily_return = np.random.normal(0.0005, 0.02)  # Slight upward trend with 2% daily volatility
                new_price = prices[-1] * (1 + daily_return)
                prices.append(max(new_price, 1))
            
            # Generate OHLC data
            data_dict[symbol] = pd.DataFrame({
                'Open': prices,
                'High': [p * np.random.uniform(1.0, 1.05) for p in prices],
                'Low': [p * np.random.uniform(0.95, 1.0) for p in prices],
                'Close': prices,
                'Volume': np.random.uniform(1000000, 10000000, len(dates)),
                'Adj Close': prices
            }, index=dates)
            
            print(f"   ✓ {symbol}: {len(data_dict[symbol])} synthetic records")

# Focus on TSLA for detailed analysis
if 'TSLA' in data_dict:
    data = data_dict['TSLA'].copy()
    print(f"\n2. Detailed Trend Analysis for TSLA...")
    
    # Flatten MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    # ============================================================================
    # ADVANCED FEATURE ENGINEERING FOR TREND DETECTION
    # ============================================================================
    
    # Basic price features
    data['Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Price_Range'] = (data['High'] - data['Low']) / data['Close']
    data['Body_Size'] = abs(data['Close'] - data['Open']) / data['Close']
    
    # Moving averages for trend detection
    for period in [5, 10, 20, 50, 100, 200]:
        data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
        data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
    
    # Technical indicators
    data['RSI'] = calculate_rsi(data['Close'])
    macd_line, signal_line, histogram = calculate_macd(data['Close'])
    data['MACD'] = macd_line
    data['MACD_Signal'] = signal_line
    data['MACD_Histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data['Close'])
    data['BB_Upper'] = bb_upper
    data['BB_Middle'] = bb_middle
    data['BB_Lower'] = bb_lower
    data['BB_Width'] = (bb_upper - bb_lower) / bb_middle
    data['BB_Position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Volatility measures
    data['Volatility_20'] = data['Return'].rolling(window=20).std()
    data['Volatility_50'] = data['Return'].rolling(window=50).std()
    data['ATR'] = calculate_atr(data, window=14)
    
    # Momentum indicators
    data['ROC'] = (data['Close'] / data['Close'].shift(10) - 1) * 100
    data['Stochastic_K'] = calculate_stochastic(data, window=14)
    data['Stochastic_D'] = data['Stochastic_K'].rolling(window=3).mean()
    
    # Volume indicators
    # Ensure Volume is a Series
    volume_series = data['Volume']
    if isinstance(volume_series, pd.DataFrame):
        volume_series = volume_series.iloc[:, 0]
    
    data['Volume_SMA'] = volume_series.rolling(window=20).mean()
    data['Volume_Ratio'] = volume_series / data['Volume_SMA']
    data['OBV'] = calculate_obv(data)
    
    # Trend detection
    trend_data = detect_trend(data)
    data = pd.concat([data, trend_data], axis=1)
    
    # Support and resistance
    sr_data = analyze_support_resistance(data)
    data = pd.concat([data, sr_data], axis=1)
    
    # Lagged features for prediction
    for lag in [1, 2, 3, 5, 10]:
        data[f'Return_Lag_{lag}'] = data['Return'].shift(lag)
        data[f'Volume_Lag_{lag}'] = data['Volume_Change'].shift(lag)
        data[f'RSI_Lag_{lag}'] = data['RSI'].shift(lag)
    
    print(f"   Created {len(data.columns)} features for trend analysis")
    
    # ============================================================================
    # TREND ANALYSIS AND VISUALIZATION
    # ============================================================================
    
    print("\n3. Trend Analysis Results...")
    
    # Current trend status
    latest = data.iloc[-1]
    print(f"   Current Price: ${latest['Close']:.2f}")
    print(f"   Trend Direction: {latest['Trend_Direction']}")
    print(f"   Trend Strength: {latest['Trend_Strength']:.2f}%")
    print(f"   RSI: {latest['RSI']:.2f}")
    print(f"   MACD: {latest['MACD']:.4f}")
    print(f"   Volume Ratio: {latest['Volume_Ratio']:.2f}")
    
    # Trend signals
    signals = []
    if latest['RSI'] > 70:
        signals.append("RSI indicates overbought (>70)")
    elif latest['RSI'] < 30:
        signals.append("RSI indicates oversold (<30)")
    
    if latest['MACD'] > latest['MACD_Signal']:
        signals.append("MACD bullish crossover")
    else:
        signals.append("MACD bearish crossover")
    
    if latest['Close'] > latest['SMA_50']:
        signals.append("Price above 50-day SMA (bullish)")
    else:
        signals.append("Price below 50-day SMA (bearish)")
    
    print(f"\n   Technical Signals:")
    for signal in signals:
        print(f"   • {signal}")
    
    # ============================================================================
    # MACHINE LEARNING FOR TREND PREDICTION
    # ============================================================================
    
    print("\n4. Machine Learning Trend Prediction...")
    
    # Prepare features for ML
    feature_columns = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Trend_Direction', 'Return']]
    data_clean = data[feature_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Target: Next day's return
    X = data_clean.iloc[:-1]  # All but last row
    y = data['Return'].iloc[1:]  # Next day's return from original data
    
    # Ensure same length
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]
    
    # Check if we have enough data
    if len(X) == 0 or len(y) == 0:
        print("   ⚠️  Not enough data for machine learning analysis")
        print("   Skipping ML prediction...")
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"   Model Performance:")
        print(f"   • R² Score: {r2:.4f}")
        print(f"   • RMSE: {np.sqrt(mse):.6f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n   Top 5 Most Important Features for Trend Prediction:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"   • {row['Feature']}: {row['Importance']:.4f}")
        
        # ============================================================================
        # TRADING RECOMMENDATIONS
        # ============================================================================
        
        print("\n5. Trading Recommendations...")
        
        # Predict next day's return
        latest_features = data_clean.iloc[-1:]
        next_day_prediction = model.predict(latest_features)[0]
        
        print(f"   Predicted Next Day Return: {next_day_prediction:.4f} ({next_day_prediction*100:.2f}%)")
        
        if next_day_prediction > 0.01:  # >1% predicted gain
            print("   🟢 BULLISH SIGNAL: Strong upward movement expected")
        elif next_day_prediction > 0.005:  # >0.5% predicted gain
            print("   🟡 MODERATE BULLISH: Slight upward movement expected")
        elif next_day_prediction < -0.01:  # >1% predicted loss
            print("   🔴 BEARISH SIGNAL: Strong downward movement expected")
        elif next_day_prediction < -0.005:  # >0.5% predicted loss
            print("   🟡 MODERATE BEARISH: Slight downward movement expected")
        else:
            print("   ⚪ NEUTRAL: Minimal movement expected")
        
        print(f"\n=== TREND ANALYSIS COMPLETE ===")
        print(f"Analyzed {len(data)} trading days")
        print(f"Current trend: {latest['Trend_Direction']}")
        print(f"Prediction confidence: {abs(r2)*100:.1f}%")
    
    # If no ML analysis was performed, still show basic completion
    if len(X) == 0 or len(y) == 0:
        print(f"\n=== TREND ANALYSIS COMPLETE ===")
        print(f"Analyzed {len(data)} trading days")
        print(f"Current trend: {latest['Trend_Direction']}")
        print("Note: ML prediction skipped due to insufficient data")

else:
    print("Error: TSLA data not available for analysis")
