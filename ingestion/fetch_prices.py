# import yfinance as yf
# df = yf.download("NKE", period="1mo")
# print(df.head())



import pandas as pd
df = pd.read_parquet("data/NKE_financials.parquet")
print(df[df["tag"] == "Revenues"].sort_values("filed_date").tail(10))

print(df.head())