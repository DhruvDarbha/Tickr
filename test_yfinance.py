"""
Simple test script to check if yfinance can pull real-time data
"""

import yfinance as yf

print("Testing yfinance data fetching...")
print("="*50)

t = yf.Ticker("MSFT")

try:
    print("\n1. Testing fast_info['last_price']...")
    price = t.fast_info["last_price"]
    print(f"   ✓ Last Price: ${price:.2f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

try:
    print("\n2. Testing fast_info (all available data)...")
    fast_info = t.fast_info
    print(f"   ✓ Fast Info available: {len(fast_info)} fields")
    if 'lastPrice' in fast_info:
        print(f"   ✓ Last Price: ${fast_info['lastPrice']:.2f}")
    if 'regularMarketPrice' in fast_info:
        print(f"   ✓ Regular Market Price: ${fast_info['regularMarketPrice']:.2f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

try:
    print("\n3. Testing history (1d period)...")
    hist = t.history(period="1d")
    if not hist.empty:
        print(f"   ✓ Got {len(hist)} records")
        print(f"   ✓ Latest Close: ${hist['Close'].iloc[-1]:.2f}")
        print(f"   ✓ Date: {hist.index[-1]}")
    else:
        print("   ✗ No data returned")
except Exception as e:
    print(f"   ✗ Failed: {e}")

try:
    print("\n4. Testing download (5d period)...")
    hist = yf.download("MSFT", period="5d", progress=False)
    if not hist.empty:
        print(f"   ✓ Got {len(hist)} records")
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)
        print(f"   ✓ Latest Close: ${hist['Close'].iloc[-1]:.2f}")
        print(f"   ✓ Date: {hist.index[-1]}")
    else:
        print("   ✗ No data returned")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "="*50)
print("Test Complete!")

