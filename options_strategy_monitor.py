import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class BitcoinTreasuryOptionsStrategy:
    """
    Comprehensive options trading strategy for Bitcoin treasury companies
    """
    
    def __init__(self):
        self.treasury_companies = {
            # Mining Companies (Strongest Performance)
            'MARA': {'type': 'mining', 'btc_holdings': 50000, 'priority': 'high'},
            'BITF': {'type': 'mining', 'btc_holdings': 1166, 'priority': 'high'},
            'CLSK': {'type': 'mining', 'btc_holdings': 12502, 'priority': 'high'},
            'RIOT': {'type': 'mining', 'btc_holdings': 19225, 'priority': 'medium'},
            'HUT': {'type': 'mining', 'btc_holdings': 10273, 'priority': 'medium'},
            'CIFR': {'type': 'mining', 'btc_holdings': 1063, 'priority': 'medium'},
            
            # Large Holders (Stability)
            'MSTR': {'type': 'holder', 'btc_holdings': 597325, 'priority': 'high'},
            'TSLA': {'type': 'holder', 'btc_holdings': 11509, 'priority': 'medium'},
            'GME': {'type': 'holder', 'btc_holdings': 4710, 'priority': 'low'},
            
            # Exchange Companies (Infrastructure)
            'COIN': {'type': 'exchange', 'btc_holdings': 9267, 'priority': 'medium'},
            'SQ': {'type': 'exchange', 'btc_holdings': 8584, 'priority': 'medium'},
            
            # Digital Asset Companies
            'BTBT': {'type': 'digital', 'btc_holdings': 418, 'priority': 'medium'},
            'EXOD': {'type': 'digital', 'btc_holdings': 2038, 'priority': 'low'},
            'SMLR': {'type': 'digital', 'btc_holdings': 4449, 'priority': 'low'}
        }
        
        self.btc_ticker = 'BTC-USD'
        self.analysis_period = 30  # days for technical analysis
        self.correlation_threshold = 0.6
        self.volatility_threshold = 0.05  # 5% daily volatility
        
    def get_current_data(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Get current market data for analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                return data
            else:
                return None
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators for options strategy"""
        if data is None or len(data) < 10:
            return {}
        
        try:
            # Basic indicators
            current_price = data['Close'].iloc[-1]
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            
            # Volatility
            daily_returns = data['Close'].pct_change()
            volatility = daily_returns.std()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # MACD
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            macd_histogram = macd - signal
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            
            # Bollinger Bands
            bb_middle = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # Volume analysis
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            return {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'volatility': volatility,
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_signal,
                'bb_position': bb_position,
                'volume_ratio': volume_ratio,
                'price_change_1d': (current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100,
                'price_change_5d': (current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100
            }
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return {}
    
    def calculate_btc_correlation(self, stock_data: pd.DataFrame, btc_data: pd.DataFrame) -> float:
        """Calculate correlation with Bitcoin price"""
        try:
            if len(stock_data) < 10 or len(btc_data) < 10:
                return 0.0
            
            # Align data lengths
            min_len = min(len(stock_data), len(btc_data))
            stock_prices = stock_data['Close'].iloc[-min_len:].reset_index(drop=True)
            btc_prices = btc_data['Close'].iloc[-min_len:].reset_index(drop=True)
            
            correlation = stock_prices.corr(btc_prices)
            return correlation if not pd.isna(correlation) else 0.0
        except:
            return 0.0
    
    def generate_options_signals(self, ticker: str, indicators: Dict, btc_correlation: float) -> Dict:
        """Generate options trading signals based on technical analysis"""
        signals = {
            'ticker': ticker,
            'current_price': indicators.get('current_price', 0),
            'signals': [],
            'recommendation': 'HOLD',
            'confidence': 0.0,
            'options_strategy': []
        }
        
        if not indicators:
            return signals
        
        # Initialize signal strength
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # Price trend signals
        if indicators.get('current_price', 0) > indicators.get('sma_20', 0):
            bullish_signals += 1
            signals['signals'].append("Price above 20-day SMA (bullish)")
        else:
            bearish_signals += 1
            signals['signals'].append("Price below 20-day SMA (bearish)")
        total_signals += 1
        
        if indicators.get('current_price', 0) > indicators.get('sma_50', 0):
            bullish_signals += 1
            signals['signals'].append("Price above 50-day SMA (bullish)")
        else:
            bearish_signals += 1
            signals['signals'].append("Price below 50-day SMA (bearish)")
        total_signals += 1
        
        # RSI signals
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            bullish_signals += 2  # Strong oversold signal
            signals['signals'].append(f"RSI oversold ({rsi:.1f}) - strong bullish")
        elif rsi < 40:
            bullish_signals += 1
            signals['signals'].append(f"RSI low ({rsi:.1f}) - bullish")
        elif rsi > 70:
            bearish_signals += 2  # Strong overbought signal
            signals['signals'].append(f"RSI overbought ({rsi:.1f}) - strong bearish")
        elif rsi > 60:
            bearish_signals += 1
            signals['signals'].append(f"RSI high ({rsi:.1f}) - bearish")
        total_signals += 2
        
        # MACD signals
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal:
            bullish_signals += 1
            signals['signals'].append("MACD bullish crossover")
        else:
            bearish_signals += 1
            signals['signals'].append("MACD bearish crossover")
        total_signals += 1
        
        # Bollinger Bands signals
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.2:
            bullish_signals += 1
            signals['signals'].append("Price near lower Bollinger Band (bullish)")
        elif bb_position > 0.8:
            bearish_signals += 1
            signals['signals'].append("Price near upper Bollinger Band (bearish)")
        total_signals += 1
        
        # Volume signals
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            if bullish_signals > bearish_signals:
                bullish_signals += 1
                signals['signals'].append("High volume supporting bullish move")
            else:
                bearish_signals += 1
                signals['signals'].append("High volume supporting bearish move")
        total_signals += 1
        
        # BTC correlation bonus
        if btc_correlation > self.correlation_threshold:
            signals['signals'].append(f"High BTC correlation ({btc_correlation:.2f})")
            # Add BTC trend influence
            if indicators.get('price_change_1d', 0) > 0:
                bullish_signals += 0.5
            else:
                bearish_signals += 0.5
        
        # Calculate recommendation
        if total_signals > 0:
            bullish_ratio = bullish_signals / total_signals
            bearish_ratio = bearish_signals / total_signals
            
            if bullish_ratio > 0.6:
                signals['recommendation'] = 'BUY'
                signals['confidence'] = bullish_ratio
            elif bearish_ratio > 0.6:
                signals['recommendation'] = 'SELL'
                signals['confidence'] = bearish_ratio
            else:
                signals['recommendation'] = 'HOLD'
                signals['confidence'] = max(bullish_ratio, bearish_ratio)
        
        # Generate options strategies
        signals['options_strategy'] = self.generate_options_strategies(
            ticker, indicators, signals['recommendation'], signals['confidence']
        )
        
        return signals
    
    def generate_options_strategies(self, ticker: str, indicators: Dict, recommendation: str, confidence: float) -> List[Dict]:
        """Generate specific options trading strategies"""
        strategies = []
        current_price = indicators.get('current_price', 0)
        volatility = indicators.get('volatility', 0)
        
        if current_price <= 0:
            return strategies
        
        # Calculate strike prices (approximate)
        atm_strike = round(current_price, 1)
        otm_strike = round(current_price * 1.05, 1)  # 5% OTM
        itm_strike = round(current_price * 0.95, 1)  # 5% ITM
        
        if recommendation == 'BUY' and confidence > 0.6:
            # Bullish strategies
            strategies.append({
                'strategy': 'Long Call',
                'description': f'Buy {ticker} call option at ${atm_strike}',
                'risk': 'Limited to premium paid',
                'reward': 'Unlimited upside',
                'timeframe': '30-45 days',
                'confidence': confidence
            })
            
            if volatility > self.volatility_threshold:
                strategies.append({
                    'strategy': 'Bull Call Spread',
                    'description': f'Buy {ticker} call at ${atm_strike}, sell call at ${otm_strike}',
                    'risk': 'Net premium paid',
                    'reward': 'Limited to spread width minus premium',
                    'timeframe': '30-45 days',
                    'confidence': confidence
                })
        
        elif recommendation == 'SELL' and confidence > 0.6:
            # Bearish strategies
            strategies.append({
                'strategy': 'Long Put',
                'description': f'Buy {ticker} put option at ${atm_strike}',
                'risk': 'Limited to premium paid',
                'reward': 'Limited to strike price minus premium',
                'timeframe': '30-45 days',
                'confidence': confidence
            })
            
            if volatility > self.volatility_threshold:
                strategies.append({
                    'strategy': 'Bear Put Spread',
                    'description': f'Buy {ticker} put at ${atm_strike}, sell put at ${itm_strike}',
                    'risk': 'Net premium paid',
                    'reward': 'Limited to spread width minus premium',
                    'timeframe': '30-45 days',
                    'confidence': confidence
                })
        
        elif recommendation == 'HOLD' and volatility > self.volatility_threshold:
            # Neutral strategies for high volatility
            strategies.append({
                'strategy': 'Iron Condor',
                'description': f'Sell {ticker} put spread and call spread around ${atm_strike}',
                'risk': 'Limited to spread width minus premium received',
                'reward': 'Limited to premium received',
                'timeframe': '30-45 days',
                'confidence': confidence
            })
        
        return strategies
    
    def run_daily_analysis(self) -> Dict:
        """Run complete daily analysis for all Bitcoin treasury companies"""
        print("🔍 Running Daily Bitcoin Treasury Options Analysis...")
        print("=" * 60)
        
        # Get BTC data for correlation
        btc_data = self.get_current_data(self.btc_ticker, self.analysis_period)
        
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'btc_price': btc_data['Close'].iloc[-1] if btc_data is not None else 0,
            'companies': {},
            'summary': {
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'high_confidence_signals': 0
            }
        }
        
        for ticker, info in self.treasury_companies.items():
            print(f"\n📊 Analyzing {ticker} ({info['type']} - {info['priority']} priority)...")
            
            # Get stock data
            stock_data = self.get_current_data(ticker, self.analysis_period)
            if stock_data is None:
                print(f"   ⚠️  No data available for {ticker}")
                continue
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(stock_data)
            if not indicators:
                print(f"   ⚠️  Could not calculate indicators for {ticker}")
                continue
            
            # Calculate BTC correlation
            btc_correlation = self.calculate_btc_correlation(stock_data, btc_data) if btc_data is not None else 0.0
            
            # Generate signals
            signals = self.generate_options_signals(ticker, indicators, btc_correlation)
            
            # Store results
            results['companies'][ticker] = {
                'info': info,
                'signals': signals,
                'indicators': indicators,
                'btc_correlation': btc_correlation
            }
            
            # Update summary
            if signals['recommendation'] == 'BUY':
                results['summary']['buy_signals'] += 1
            elif signals['recommendation'] == 'SELL':
                results['summary']['sell_signals'] += 1
            else:
                results['summary']['hold_signals'] += 1
            
            if signals['confidence'] > 0.7:
                results['summary']['high_confidence_signals'] += 1
            
            # Print results
            print(f"   💰 Current Price: ${signals['current_price']:.2f}")
            print(f"   📈 Recommendation: {signals['recommendation']} (Confidence: {signals['confidence']:.1%})")
            print(f"   🔗 BTC Correlation: {btc_correlation:.2f}")
            
            if signals['options_strategy']:
                print(f"   🎯 Options Strategy: {signals['options_strategy'][0]['strategy']}")
        
        return results
    
    def generate_daily_report(self, results: Dict) -> str:
        """Generate a comprehensive daily report"""
        report = f"""
📋 BITCOIN TREASURY OPTIONS DAILY REPORT
{'=' * 60}
Generated: {results['timestamp']}
BTC Price: ${results['btc_price']:,.2f}

📊 SUMMARY STATISTICS:
   • Companies Analyzed: {len(results['companies'])}
   • BUY Signals: {results['summary']['buy_signals']}
   • SELL Signals: {results['summary']['sell_signals']}
   • HOLD Signals: {results['summary']['hold_signals']}
   • High Confidence Signals: {results['summary']['high_confidence_signals']}

🎯 TOP RECOMMENDATIONS:
"""
        
        # Sort companies by confidence and priority
        recommendations = []
        for ticker, data in results['companies'].items():
            signals = data['signals']
            info = data['info']
            
            if signals['confidence'] > 0.5:  # Only include confident signals
                priority_score = {'high': 3, 'medium': 2, 'low': 1}[info['priority']]
                recommendations.append({
                    'ticker': ticker,
                    'recommendation': signals['recommendation'],
                    'confidence': signals['confidence'],
                    'priority': info['priority'],
                    'priority_score': priority_score,
                    'type': info['type'],
                    'current_price': signals['current_price']
                })
        
        # Sort by priority score and confidence
        recommendations.sort(key=lambda x: (x['priority_score'], x['confidence']), reverse=True)
        
        for i, rec in enumerate(recommendations[:5], 1):
            report += f"""
{i}. {rec['ticker']} ({rec['type'].upper()}) - {rec['recommendation']}
   • Confidence: {rec['confidence']:.1%}
   • Priority: {rec['priority'].upper()}
   • Current Price: ${rec['current_price']:.2f}
"""
        
        report += f"""
💡 STRATEGIC INSIGHTS:
   • Focus on high-priority mining companies (MARA, BITF, CLSK)
   • Monitor MSTR for stability and large position influence
   • Use BTC correlation for timing entry/exit points
   • Consider volatility for options strategy selection

🚨 RISK MANAGEMENT:
   • Set stop-loss orders for all positions
   • Diversify across different company types
   • Monitor BTC price movements closely
   • Avoid over-leveraging in high volatility periods

📅 NEXT UPDATE: Tomorrow at market open
"""
        
        return report

def main():
    """Main function to run the daily analysis"""
    strategy = BitcoinTreasuryOptionsStrategy()
    
    # Run daily analysis
    results = strategy.run_daily_analysis()
    
    # Generate and print report
    report = strategy.generate_daily_report(results)
    print(report)
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"options_analysis_{timestamp}.json"
    
    # Convert results to JSON-serializable format
    json_results = {
        'timestamp': results['timestamp'],
        'btc_price': float(results['btc_price']),
        'summary': results['summary'],
        'companies': {}
    }
    
    for ticker, data in results['companies'].items():
        json_results['companies'][ticker] = {
            'info': data['info'],
            'signals': {
                'ticker': data['signals']['ticker'],
                'current_price': float(data['signals']['current_price']),
                'recommendation': data['signals']['recommendation'],
                'confidence': float(data['signals']['confidence']),
                'signals': data['signals']['signals'],
                'options_strategy': data['signals']['options_strategy']
            },
            'indicators': {k: float(v) if isinstance(v, (int, float)) else v 
                          for k, v in data['indicators'].items()},
            'btc_correlation': float(data['btc_correlation'])
        }
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"💾 Analysis saved to {filename}")

if __name__ == "__main__":
    main() 