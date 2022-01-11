import pandas as pd
from pandas_datareader import data as wb

companies = pd.read_csv("Data\constituents_csv.csv")
tickers = companies.Symbol
ticker_data=wb.DataReader(tickers,start='2000-1-1',data_source='yahoo')

ticker_data.to_csv(r'Data\stock_data.csv',sep=';',float_format='%.2f')

