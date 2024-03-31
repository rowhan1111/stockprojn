import yahoo_fin.stock_info as si
import time
import yfinance as yf
import pandas as pd


class StocksData:
    def __init__(self, start_date):
        self.ticker_list = si.tickers_nasdaq()
        self.start_date = start_date

    def stock_to_pd(self):
        return_dict = {}
        for to_use in self.ticker_list:
            return_dict[to_use] = yf.download(to_use, start=self.start_date)
        return return_dict


#print(yf.download('aapl'))
test = yf.Ticker('aapl')
print(test)
#print(test.balance_sheet)
#print(pd.concat([test.financials, test.balance_sheet]))
#print(si.get_earnings_history('AAPL'))
#print(test.dividends)
#test.get_earnings().to_csv("earnings.csv", index=True)
#hist = test.history(period='max', interval="1d")
#hist.to_csv("test1.csv", index=True)

#print(test.news)
#print(si.tickers_nasdaq())
#test = si.tickers_nasdaq()[0]
#print(si.get_data(test))
#print(si.get_cash_flow("tsla"))
#si.get_earnings("aapl").to_csv("test2.csv", index=True)
'''
for i in si.tickers_nasdaq():
    #get_analysts_info, get_balance_sheet, get_cash_flow, get_data, get_dividends, get_earnings_history
    #balance sheet, cash flow, income statement => get_financials
    print(i)
    print(si.get_financials(i))
'''