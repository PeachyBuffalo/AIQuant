from polygon import RESTClient
from datetime import datetime
import os

API_KEY = "API_KEY" # Replace with your Polygon API key
client = RESTClient(API_KEY) # Initialize the client

def get_previous_close(symbol: str): # Get the previous close price for a given symbol for free tier. Limited to 5 pulls per minute.
    try:
        close = client.get_previous_close(symbol)
        print(f"[{symbol}] Previous close: ${close.results[0].c}")
    except Exception as e:
        print(f"Error fetching previous close for {symbol}: {e}")

def get_daily_aggregates(symbol: str, from_date: str, to_date: str):
    try:
        aggs = client.list_aggs(symbol, 1, "day", from_date, to_date, limit=10)
        for agg in aggs:
            date_str = datetime.utcfromtimestamp(agg.timestamp / 1000).strftime('%Y-%m-%d')
            print(f"{symbol} - {date_str}: Open={agg.open}, High={agg.high}, Low={agg.low}, Close={agg.close}")
    except Exception as e:
        print(f"Error fetching aggregates for {symbol}: {e}")

if __name__ == "__main__":
    get_previous_close("AAPL")
    get_daily_aggregates("AAPL", "2024-01-01", "2024-01-10")