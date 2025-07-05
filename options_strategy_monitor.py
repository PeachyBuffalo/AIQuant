import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
from typing import Dict, List, Tuple
import warnings
import subprocess
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TREASURY_COMPANIES = [
    # Mining Companies (Strongest Performance)
    'MARA', 'BITF', 'CLSK', 'RIOT', 'HUT', 'CIFR',
    # Large Holders (Stability)
    'MSTR', 'TSLA', 'GME',
    # Exchange Companies (Infrastructure)
    'COIN', 'SQ',
    # Digital Asset Companies
    'BTBT', 'EXOD', 'SMLR'
]
BTC_TICKER = 'BTC-USD'
ANALYSIS_DAYS = 60
MIN_DATA = 30
CORR_WINDOW = 30
VOLATILITY_THRESHOLD = 0.05

# --- NOTIFICATION FUNCTION ---
def send_mac_notification(title, message):
    try:
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(["osascript", "-e", script])
    except Exception as e:
        print(f"Notification error: {e}")

# --- INDICATOR FUNCTIONS ---
def calculate_indicators(data):
    if data is None or len(data) < MIN_DATA:
        return None
    close = data['Close']
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    # Use float values and check for NaN
    last_close = float(close.iloc[-1])
    last_sma20 = float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1].item()) else None
    last_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1].item()) else None
    trend = 'side'
    if last_sma20 is not None and last_sma50 is not None:
        if last_close > last_sma20 and last_close > last_sma50:
            trend = 'up'
        elif last_close < last_sma20 and last_close < last_sma50:
            trend = 'down'
    volatility = float(close.pct_change().std())
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return {
        'trend': trend,
        'volatility': volatility,
        'rsi': float(rsi.iloc[-1]),
        'macd': float(macd.iloc[-1]),
        'macd_signal': float(signal.iloc[-1]),
        'sma20': last_sma20 if last_sma20 is not None else 0.0,
        'sma50': last_sma50 if last_sma50 is not None else 0.0,
        'current_price': last_close,
        'price_change_1d': float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0.0
    }

def get_btc_correlation(stock_data, btc_data):
    if stock_data is None or btc_data is None:
        return np.nan
    min_len = min(len(stock_data), len(btc_data))
    if min_len < CORR_WINDOW:
        return np.nan
    # Ensure Series, not DataFrame
    stock_close = stock_data['Close'].iloc[-CORR_WINDOW:]
    if isinstance(stock_close, pd.DataFrame):
        stock_close = stock_close.iloc[:, 0]
    btc_close = btc_data['Close'].iloc[-CORR_WINDOW:]
    if isinstance(btc_close, pd.DataFrame):
        btc_close = btc_close.iloc[:, 0]
    stock_close = stock_close.reset_index(drop=True)
    btc_close = btc_close.reset_index(drop=True)
    return stock_close.corr(btc_close)

# --- STRATEGY LOGIC ---
def recommend_option(ind, btc_corr):
    if ind is None:
        return ('HOLD', 'Insufficient data', 0)
    confidence = 0
    if ind['trend'] == 'up' and ind['macd'] > ind['macd_signal'] and ind['rsi'] < 70:
        confidence += 1
    if ind['trend'] == 'down' and ind['macd'] < ind['macd_signal'] and ind['rsi'] > 30:
        confidence -= 1
    if btc_corr is not None and not np.isnan(btc_corr):
        if btc_corr > 0.6:
            confidence += 0.5
        elif btc_corr < -0.6:
            confidence -= 0.5
    # Volatility filter
    if ind['volatility'] > VOLATILITY_THRESHOLD:
        vol_note = 'High volatility'
    else:
        vol_note = 'Normal volatility'
    # Decision
    if confidence >= 1:
        return ('BUY CALL', f'Bullish ({vol_note})', confidence)
    elif confidence <= -1:
        return ('BUY PUT', f'Bearish ({vol_note})', -confidence)
    else:
        return ('HOLD', f'Neutral ({vol_note})', abs(confidence))

# --- MAIN SCRIPT ---
def main():
    print(f"\n=== Bitcoin Treasury Companies Options Monitor ===\nDate: {datetime.now().strftime('%Y-%m-%d')}\n")
    btc_data = yf.download(BTC_TICKER, period=f'{ANALYSIS_DAYS}d', progress=False)
    summary = []
    for ticker in TREASURY_COMPANIES:
        data = yf.download(ticker, period=f'{ANALYSIS_DAYS}d', progress=False)
        ind = calculate_indicators(data)
        btc_corr = get_btc_correlation(data, btc_data)
        rec, note, conf = recommend_option(ind, btc_corr)
        summary.append({
            'Ticker': ticker,
            'Trend': ind['trend'] if ind else 'N/A',
            'BTC Corr': f"{btc_corr:.2f}" if btc_corr is not None and not np.isnan(btc_corr) else 'N/A',
            'Signal': rec,
            'Note': note,
            'Confidence': f"{conf:.2f}",
            'Price': f"${ind['current_price']:.2f}" if ind else 'N/A'
        })
    # Print summary table
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    # Print recommendations
    print("\n--- Strategy Recommendations ---")
    top_msgs = []
    for row in summary:
        if row['Signal'] == 'BUY CALL':
            msg = f"[BUY CALL] {row['Ticker']} at {row['Price']} | {row['Note']} | BTC Corr: {row['BTC Corr']} | Confidence: {row['Confidence']}"
            print(msg)
            top_msgs.append(msg)
        elif row['Signal'] == 'BUY PUT':
            msg = f"[BUY PUT] {row['Ticker']} at {row['Price']} | {row['Note']} | BTC Corr: {row['BTC Corr']} | Confidence: {row['Confidence']}"
            print(msg)
            top_msgs.append(msg)
    print("\nStrategic Insights:")
    print("- Focus on Mining Companies: MARA, BITF, CLSK showing strongest performance")
    print("- Monitor Large Holders: MSTR's massive position provides stability")
    print("- Diversify Exposure: Mix of mining, exchange, and holding companies")
    print("- Timing Strategy: Use correlation with BTC price for entry/exit points")
    print("\nRisk Management:")
    print("- Set stop-loss orders for all positions")
    print("- Diversify across company types")
    print("- Monitor BTC price movements closely")
    print("- Avoid over-leveraging in high volatility periods")

    # Send macOS notification with top recommendations
    if top_msgs:
        notif_msg = '\n'.join(top_msgs[:2])  # Show up to 2 in notification
        send_mac_notification(
            title="BTC Treasury Options Signals",
            message=notif_msg
        )
    else:
        send_mac_notification(
            title="BTC Treasury Options Signals",
            message="No strong buy/sell signals today."
        )

if __name__ == "__main__":
    main() 