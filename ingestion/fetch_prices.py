# import yfinance as yf
# df = yf.download("NKE", period="1mo")
# print(df.head())



import pandas as pd
fin = pd.read_parquet("data/NKE_financials.parquet")
print(fin[fin["tag"] == "Revenues"].sort_values("filed_date").tail(5))
