import yfinance as yf
df = yf.download("NKE", period="1mo")
print(df.head())

