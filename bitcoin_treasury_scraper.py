import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from datetime import datetime
import re

def scrape_bitcoin_treasuries():
    """
    Scrape Bitcoin treasury data from bitcointreasuries.net
    """
    print("🔍 Fetching updated Bitcoin treasury data...")
    
    try:
        # Try to fetch from bitcointreasuries.net
        url = "https://bitcointreasuries.net/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for the entities table data
        treasury_data = []
        
        # Try to find table data
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:
                    try:
                        # Extract text and clean it
                        company = cells[0].get_text(strip=True)
                        ticker = cells[1].get_text(strip=True)
                        btc_holdings = cells[2].get_text(strip=True)
                        value = cells[3].get_text(strip=True)
                        
                        # Clean ticker - remove country flags and invalid characters
                        ticker = clean_ticker(ticker)
                        
                        # Only add if we have valid data
                        if ticker and len(ticker) <= 10 and not ticker.startswith('🇺🇸') and not ticker.startswith('🇨🇦'):
                            treasury_data.append({
                                'Company': company,
                                'Ticker': ticker,
                                'BTC_Holdings': btc_holdings,
                                'Value_USD': value
                            })
                    except:
                        continue
        
        if treasury_data:
            df = pd.DataFrame(treasury_data)
            print(f"✅ Successfully scraped {len(df)} Bitcoin treasury companies")
            return df
        else:
            print("⚠️  No table data found, using fallback data")
            return get_fallback_treasury_data()
            
    except Exception as e:
        print(f"❌ Error scraping data: {e}")
        print("📋 Using fallback Bitcoin treasury data...")
        return get_fallback_treasury_data()

def clean_ticker(ticker):
    """
    Clean ticker symbol by removing country flags and invalid characters
    """
    if not ticker:
        return ""
    
    # Remove country flags and emojis
    # Remove emoji patterns
    ticker = re.sub(r'[🇺🇸🇨🇦🇯🇵🇩🇪🇭🇰🇰🇾🇫🇷🇸🇬🇳🇴🇧🇷🇦🇷🇲🇹🇹🇭🇹🇷🇰🇷🇨🇳🇬🇧🇦🇺🇸🇪🇯🇪🇦🇪🇮🇳🇮🇹🇧🇭🇿🇦]', '', ticker)
    
    # Remove extra whitespace and common invalid characters
    ticker = ticker.strip()
    ticker = re.sub(r'[^\w\.-]', '', ticker)  # Keep only alphanumeric, dots, and hyphens
    
    return ticker

def get_fallback_treasury_data():
    """
    Fallback data with known Bitcoin treasury companies
    """
    treasury_data = [
        {'Company': 'Microstrategy, Inc.', 'Ticker': 'MSTR', 'BTC_Holdings': '597,325', 'Value_USD': '$25.2B'},
        {'Company': 'Marathon Digital Holdings', 'Ticker': 'MARA', 'BTC_Holdings': '50,000', 'Value_USD': '$2.1B'},
        {'Company': 'Riot Platforms, Inc.', 'Ticker': 'RIOT', 'BTC_Holdings': '19,225', 'Value_USD': '$813M'},
        {'Company': 'Tesla, Inc.', 'Ticker': 'TSLA', 'BTC_Holdings': '11,509', 'Value_USD': '$487M'},
        {'Company': 'Hut 8 Mining Corp', 'Ticker': 'HUT', 'BTC_Holdings': '10,273', 'Value_USD': '$434M'},
        {'Company': 'Coinbase Global, Inc.', 'Ticker': 'COIN', 'BTC_Holdings': '9,267', 'Value_USD': '$392M'},
        {'Company': 'CleanSpark, Inc.', 'Ticker': 'CLSK', 'BTC_Holdings': '12,502', 'Value_USD': '$529M'},
        {'Company': 'Bitfarms Ltd.', 'Ticker': 'BITF', 'BTC_Holdings': '1,166', 'Value_USD': '$49M'},
        {'Company': 'Cipher Mining', 'Ticker': 'CIFR', 'BTC_Holdings': '1,063', 'Value_USD': '$45M'},
        {'Company': 'Bit Digital, Inc.', 'Ticker': 'BTBT', 'BTC_Holdings': '418', 'Value_USD': '$18M'},
        {'Company': 'GameStop Corp.', 'Ticker': 'GME', 'BTC_Holdings': '4,710', 'Value_USD': '$199M'},
        {'Company': 'Semler Scientific', 'Ticker': 'SMLR', 'BTC_Holdings': '4,449', 'Value_USD': '$188M'},
        {'Company': 'Exodus Movement, Inc', 'Ticker': 'EXOD', 'BTC_Holdings': '2,038', 'Value_USD': '$86M'},
        {'Company': 'Block, Inc.', 'Ticker': 'SQ', 'BTC_Holdings': '8,584', 'Value_USD': '$363M'},
        {'Company': 'MercadoLibre, Inc.', 'Ticker': 'MELI', 'BTC_Holdings': '570', 'Value_USD': '$24M'}
    ]
    
    df = pd.DataFrame(treasury_data)
    print(f"📋 Using fallback data with {len(df)} companies")
    return df

def get_treasury_tickers():
    """
    Get list of tickers for Bitcoin treasury companies
    """
    df = scrape_bitcoin_treasuries()
    tickers = df['Ticker'].tolist()
    
    # Filter out any invalid tickers and add common suffixes
    valid_tickers = []
    for ticker in tickers:
        if ticker and len(ticker) > 0:
            # Add common suffixes if not present
            if '.' not in ticker and len(ticker) <= 5:
                valid_tickers.append(ticker)
    
    return valid_tickers

def analyze_treasury_performance(treasury_df, price_data_dict):
    """
    Analyze performance of Bitcoin treasury companies
    """
    print("\n🏦 Bitcoin Treasury Performance Analysis:")
    
    performance_data = []
    
    for _, row in treasury_df.iterrows():
        ticker = row['Ticker']
        btc_holdings = row['BTC_Holdings']
        
        if ticker in price_data_dict:
            data = price_data_dict[ticker]
            if len(data) > 0:
                current_price = data['Close'].iloc[-1]
                start_price = data['Close'].iloc[0]
                price_change = ((current_price - start_price) / start_price) * 100
                volatility = data['Close'].pct_change().std() * 100
                
                performance_data.append({
                    'Ticker': ticker,
                    'Company': row['Company'],
                    'BTC_Holdings': btc_holdings,
                    'Current_Price': current_price,
                    'Price_Change_%': price_change,
                    'Volatility_%': volatility
                })
    
    if performance_data:
        perf_df = pd.DataFrame(performance_data)
        perf_df = perf_df.sort_values('Price_Change_%', ascending=False)
        
        print(f"\n📊 Treasury Performance Summary ({len(perf_df)} companies):")
        print(f"   • Average Return: {perf_df['Price_Change_%'].mean():.2f}%")
        print(f"   • Average Volatility: {perf_df['Volatility_%'].mean():.2f}%")
        
        print(f"\n🏆 Top 5 Treasury Performers:")
        for _, row in perf_df.head(5).iterrows():
            print(f"   • {row['Ticker']}: {row['Price_Change_%']:.2f}% (${row['Current_Price']:.2f}) - {row['BTC_Holdings']} BTC")
        
        return perf_df
    
    return None

if __name__ == "__main__":
    # Test the scraper
    treasury_df = scrape_bitcoin_treasuries()
    print(f"\n📋 Bitcoin Treasury Data:")
    print(treasury_df.head(10))
    
    # Get tickers for analysis
    tickers = get_treasury_tickers()
    print(f"\n🎯 Treasury Tickers: {tickers}") 