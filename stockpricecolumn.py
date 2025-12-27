import pandas as pd
import yfinance as yf

csv_path = "/Users/shivvinodshankar/Tickr/MSFT_all_dates_data.csv"
ticker = "MSFT"

df = pd.read_csv(csv_path)

import pandas as pd
import yfinance as yf

csv_path = "/Users/shivvinodshankar/Tickr/MSFT_all_dates_data.csv"
ticker = "MSFT"

df = pd.read_csv(csv_path)

# Choose which date to price on:
# Usually: published_at (when the article was published)
df["price_date"] = pd.to_datetime(df["published_at"], utc=True).dt.tz_convert(None).dt.normalize()
# If you prefer date_requested instead, swap the line above for:
# df["price_date"] = pd.to_datetime(df["date_requested"]).dt.normalize()

out = []
for t, g in df.groupby("ticker"):
    g = g.sort_values("price_date").copy()

    start = g["price_date"].min()
    end = g["price_date"].max() + pd.Timedelta(days=1)

    hist = yf.download(t, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    
    # Handle MultiIndex from yfinance
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.droplevel(1)
    
    hist = hist.reset_index()
    hist["price_date"] = pd.to_datetime(hist["Date"]).dt.normalize()

    # previous trading day match for non-trading days
    hist = hist[["price_date", "Close"]].sort_values("price_date")
    g = pd.merge_asof(g, hist, on="price_date", direction="backward")

    g = g.rename(columns={"Close": "close_price_on_price_date"})
    out.append(g)

df2 = pd.concat(out, ignore_index=True)

# Save back to CSV
df2.to_csv("with_prices.csv", index=False)
print(f"Saved {len(df2)} rows with prices to with_prices.csv")
print(f"Columns: {list(df2.columns)}")
