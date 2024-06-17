import yahoo_fin.stock_info as si
import yfinance as yf
import requests
import time
import pandas as pd
from datetime import date
import datetime
import os
from dotenv import load_dotenv
# may need modification in sorting news data to separate news written before, during & after market opening
from tqdm import tqdm
from pandas_datareader import data as pdr
import numpy as np


class DataCreator:
    def __init__(self, tickers_list, start_date, end_date=(datetime.datetime.today()
                                                           - datetime.timedelta(days=1)).strftime("%Y-%m-%d")):
        # start_date => ex. 2023-01-22
        self.tickers_list = tickers_list
        self.funds_list = []
        self.not_available = []
        self.tickers_obj_list = [yf.Ticker(i) for i in tickers_list]
        self.all_dict = {}
        self.start_date = start_date
        self.end_date = end_date
        load_dotenv()
        api_key = os.environ['api_key']
        api_secret = os.environ['api_secret']
        self.headers = {
            'Apca-Api-Key-Id': api_key,
            'Apca-Api-Secret-Key': api_secret
        }
        # set header for url to retrieve news headlines with

    def add_to_all_dict(self, news):
        # create dictionary to store headlines and sources for all tickers in nasdaq
        for single_news in news:
            for sing_ticker in list(set(single_news['symbols']).intersection(set(self.tickers_list))):
                temp_date = single_news['updated_at'][:10]
                if temp_date not in self.all_dict[sing_ticker].keys():
                    self.all_dict[sing_ticker][temp_date] = {'headline': [single_news['headline']],
                                                             'source': [single_news['source']]}
                else:
                    self.all_dict[sing_ticker][temp_date]['headline'].append(single_news['headline'])
                    self.all_dict[sing_ticker][temp_date]['source'].append(single_news['source'])

    # function to convert news to pandas dataframe
    def news_to_pd(self):
        # make pd dataframes containing headlines & dates
        self.all_dict = {i: {datetime.datetime.strftime(f, "%Y-%m-%d"): {'headline': [], 'source': []}
                             for f in pd.date_range(start=self.start_date, end=self.end_date, freq='D')}
                         for i in self.tickers_list}
        tickers_for_link = '%2C'.join(self.tickers_list)
        # set start and ending time
        start_time = self.start_date + "T00%3A00%3A00Z"
        end_time = self.end_date + "T23%3A59%3A59Z"
        url = (f'https://data.alpaca.markets/v1beta1/news?&sort=asc&symbols={tickers_for_link}'
               f'&limit=50&start={start_time}&end={end_time}')
        response = requests.get(url, headers=self.headers).json()  # retrieve response from alpaca api
        self.add_to_all_dict(response['news'])  # add news to the dictionary
        total_iterations = None
        curr_it = 0
        progress_bar = tqdm(total=total_iterations, desc="Collecting News")  # no given max sequence, so for tracking purposes
        # iterate over all given pages until the last page is reached
        while response['next_page_token']:
            url2 = url + f"&page_token={response['next_page_token']}"
            response = requests.get(url2, headers=self.headers).json()
            self.add_to_all_dict(response['news'])
            curr_it += 1
            progress_bar.update(1)
        progress_bar.close()
        return_dict = {}
        for ticker, item in tqdm(self.all_dict.items(), desc="Processing News"):
            return_dict[ticker] = pd.DataFrame.from_dict(item,
                                                         orient='index', columns=['headline', 'source'])
            return_dict[ticker].index = pd.to_datetime(return_dict[ticker].index)
        return return_dict

    # function to get stock data (prices at given dates)
    def stock_to_pd(self):
        # override yahoo finance to return pandas results
        yf.pdr_override()
        # get data for all the required tickers
        yahoo_all_data = pdr.get_data_yahoo(self.tickers_list, start=self.start_date, end=self.end_date)
        return_dict = {i: pd.DataFrame() for i in self.tickers_list}
        # process stock data to be put into a csv
        for col, ticker in tqdm(yahoo_all_data.columns, desc="Processing Stock Data"):
            if ticker in self.not_available:
                continue
            pd_data = yahoo_all_data[col][ticker]
            checker = pd_data.loc[pd_data.index[-10:-1]]
            if not all(np.isnan(value) for value in checker):
                return_dict[ticker][col] = pd_data
            else:
                if ticker in self.tickers_list:
                    self.not_available.append(ticker)
                    self.tickers_list.remove(ticker)
        self.tickers_obj_list = [yf.Ticker(i) for i in self.tickers_list]
        return return_dict

    # function to get financial data from yf
    def get_financial(self):
        return_dict = {}
        # get financial data for given ticker
        for to_use in tqdm(self.tickers_obj_list, desc="Financial Data"):
            balance = to_use.quarterly_balancesheet.transpose()
            date_list = balance.index
            temp_dict = {}
            start_time = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
            for time_index in range(0, len(date_list)):
                end_date = date_list[time_index-1] if time_index != 0 else date.today()
                time_to_use = date_list[time_index]
                if time_to_use <= start_time:
                    for i in pd.date_range(start=start_time, end=end_date, freq='D'):
                        temp_dict[i.to_pydatetime()] = balance.loc[time_to_use]
                    break
                elif time_to_use > start_time:
                    for i in pd.date_range(start=time_to_use, end=end_date, freq='D'):
                        temp_dict[i.to_pydatetime()] = balance.loc[time_to_use]
            return_dict[to_use.ticker] = pd.DataFrame.from_dict(temp_dict).transpose()
        return return_dict

    # function to get quarterly financial data for the tickers
    def get_financial_quarters(self):
        return_dict = {}
        for to_use in tqdm(self.tickers_obj_list, desc="quarter datas"):
            balance = to_use.quarterly_balancesheet
            return_dict[to_use.ticker] = balance
        return return_dict

    # function to get yearly data for the tickers
    def get_yearly_n_info(self):
        yearly_dict = {}
        info_list = ['sector', 'industry']
        sub_info_list = ['category', 'legalType']
        info_dict = {}
        sub_info_dict = {}
        for yf_ticker in tqdm(self.tickers_obj_list, desc='yearly data'):
            yearly_datas = pd.concat([yf_ticker.balance_sheet, yf_ticker.financials, yf_ticker.income_stmt])
            ticker_info = yf_ticker.info
            try:
                info_dict[yf_ticker.ticker] = [ticker_info[i] for i in info_list]
            except:
                try:
                    sub_info_dict[yf_ticker.ticker] = [ticker_info[i] for i in sub_info_list]
                    self.funds_list.append(yf_ticker.ticker)
                except:
                    self.not_available.append(yf_ticker.ticker)
                    continue

            yearly_datas = yearly_datas.rename(columns={i: datetime.datetime.strftime(i, "%Y-%m-%d") + yf_ticker.ticker
                                                        for i in yearly_datas.columns})
            yearly_dict[yf_ticker.ticker] = yearly_datas
        return (yearly_dict, pd.DataFrame.from_dict(info_dict, orient='index', columns=info_list),
                pd.DataFrame.from_dict(sub_info_dict, orient='index', columns=sub_info_list))

    # function to create a csv for a given ticker with all the information from various sources
    def create_csv(self, file_path):
        stocks_data = self.stock_to_pd()
        news_data = self.news_to_pd()
        # financial_data = self.get_financial()
        yearly, info, sub_info = self.get_yearly_n_info()
        quarter_info = self.get_financial_quarters()
        info.to_csv(file_path + "info.csv")
        sub_info.to_csv(file_path + "sub/info.csv")
        for ticker in list(set(self.tickers_list).difference(set(self.funds_list)).difference(set(self.not_available))):
            news_data[ticker].join(stocks_data[ticker]).to_csv(file_path + ticker + ".csv")
            yearly[ticker].to_csv(f"{file_path}yearly_{ticker}.csv")
            quarter_info[ticker].to_csv(f"{file_path}quarterly_{ticker}.csv")
        for ticker in self.funds_list:
            news_data[ticker].join(stocks_data[ticker]).to_csv(file_path + "sub/" + ticker + ".csv")
            yearly[ticker].to_csv(f"{file_path}sub/yearly_{ticker}.csv")
            quarter_info[ticker].to_csv(f"{file_path}sub/quarterly_{ticker}.csv")

if __name__ == '__main__':
    # si.tickers_nasdaq()
    # ['AAPL', 'TSLA']
    invalid_tickers = ['GLACR', 'GLLIR', 'GLSTR', 'GODNR', 'HHGCR', 'HOVRW', 'HSPOR', 'IGTAR', 'IMAQR', 'KACLR',
                    'KYCHR', 'LBBBR', 'MARXR', 'MBTCR', 'MCACR', 'MCAFR', 'MCAGR', 'MSSAR', 'NNAGR', 'NOVVR',
                    'NVACR', 'PLTNR', 'PPHPR', 'QETAR', 'QOMOR', 'RFACR', 'RWODR', 'SAGAR', 'STRCW', 'SVIIR',
                    'TENKR', 'TMTCR', 'WINVR', 'WTMAR', 'YOTAR', 'ZAZZT', 'ZBZZT', 'ZCZZT', 'ZJZZT', 'ZVZZT',
                    'ZWZZT', 'ZXYZ.A', 'ZXZZT']
    invalid_tickers = ['AACIW', 'AAGRW', 'ABLLW', 'ABLVW', 'ACABW', 'ACACW', 'ACAHW', 'ACBAW', 'ACONW', 'ADNWW', 'ADOCR', 'ADOCW', 'ADSEW', 'ADTHW', 'ADVWW', 'AEAEW', 'AENTW', 'AERTW', 'AFARW', 'AFRIW', 'AGBAW', 'AGRIW', 'AIBBR', 'AIMAW', 'AIMDW', 'AISPW', 'AITRR', 'ALCYW', 'ALSAR', 'ALSAW', 'ALVOW', 'ANGHW', 'ANSCW', 'AOGOW', 'AONCW', 'APACW', 'APCXW', 'APTMW', 'APXIW', 'AQUNR', 'ARBEW', 'AREBW', 'ARIZR', 'ARIZW', 'ARKOW', 'ARRWW', 'ARTLW', 'ASCAR', 'ASCAW', 'ASCBR', 'ASCBW', 'ASTLW', 'ASTSW', 'ATAKR', 'ATMCR', 'ATMCW', 'ATMVR', 'ATNFW', 'AUROW', 'AUUDW', 'AVHIW', 'AVPTW', 'AWINW', 'BAERW', 'BAYAR', 'BCDAW', 'BCSAW', 'BCTXW', 'BEATW', 'BEEMW', 'BENFW', 'BETRW', 'BFIIW', 'BFRGW', 'BFRIW', 'BHACW', 'BIAFW', 'BLACR', 'BLACW', 'BLDEW', 'BLEUR', 'BLEUW', 'BNIXR', 'BNIXW', 'BNZIW', 'BOCNW', 'BOWNR', 'BRACR', 'BREZR', 'BREZW', 'BRKHW', 'BROGW', 'BRSHW', 'BTBDW', 'BTCTW', 'BTMWW', 'BUJAR', 'BUJAW', 'BWAQR', 'BWAQW', 'BYNOW', 'BZFDW', 'CAPTW', 'CCTSW', 'CDAQW', 'CDIOW', 'CDROW', 'CDTTW', 'CEADW', 'CELUW', 'CETUR', 'CETUW', 'CFFSW', 'CIFRW', 'CINGW', 'CITEW', 'CLBTW', 'CLNNW', 'CLOER', 'CLRCR', 'CLRCW', 'CMAXW', 'CMCAW', 'CMPOW', 'CNFRZ', 'CNGLW', 'COCHW', 'COEPW', 'COMSW', 'CONXW', 'CPTNW', 'CRESW', 'CREVW', 'CRGOW', 'CSLMR', 'CSLMW', 'CSLRW', 'CSSEL', 'CTCXW', 'CURIW', 'CXAIW', 'DAVEW', 'DBGIW', 'DCFCW', 'DECAW', 'DFLIW', 'DHACW', 'DHCAW', 'DHCNL', 'DISTR', 'DISTW', 'DMAQR', 'DPCSW', 'DRMAW', 'DRTSW', 'DTSTW', 'DUETW', 'DWACW', 'EACPW', 'ECDAW', 'ECXWW', 'EDBLW', 'EFTRW', 'EMCGR', 'EMCGW', 'EMLDW', 'ENCPW', 'ENGNW', 'ESACW', 'ESGLW', 'ESHAR', 'ESLAW', 'EUDAW', 'EVGRW', 'EVLVW', 'FATBW', 'FAZEW', 'FBYDW', 'FEXDR', 'FEXDW', 'FFIEW', 'FGIWW', 'FHLTW', 'FIACW', 'FICVW', 'FLFVR', 'FLFVW', 'FMSTW', 'FNVTW', 'FORLW', 'FREEW', 'FTIIW', 'GAMCW', 'GBBKR', 'GBBKW', 'GCMGW', 'GDEVW', 'GDSTR', 'GDSTW', 'GECCZ', 'GFAIW', 'GGROW', 'GHIXW', 'GIPRW', 'GLACR', 'GLLIR', 'GLLIW', 'GLSTR', 'GLSTW', 'GMBLW', 'GMBLZ', 'GMFIW', 'GODNR', 'GOEVW', 'GORV', 'GOVXW', 'GPACW', 'GROMW', 'GRRRW', 'GSDWW', 'GSMGW', 'GTACW', 'HAIAW', 'HCMAW', 'HCVIW', 'HGASW', 'HHGCR', 'HHGCW', 'HOFVW', 'HOLOW', 'HOVR', 'HOVRW', 'HSCSW', 'HSPOR', 'HSPOW', 'HTZWW', 'HUBCW', 'HUBCZ', 'HUMAW', 'HYMCW', 'HYZNW', 'ICUCW', 'IGTAR', 'IGTAW', 'IMACW', 'IMAQR', 'IMAQW', 'IMTXW', 'INTEW', 'INVZW', 'IPXXW', 'IRAAW', 'ISRLW', 'IVCAW', 'IVCBW', 'IVCPW', 'IVDAW', 'IXAQW', 'JFBRW', 'JSPRW', 'JTAIW', 'KACLR', 'KACLW', 'KERNW', 'KITTW', 'KPLTW', 'KRNLW', 'KTTAW', 'KVACW', 'KWESW', 'KYCHR', 'KYCHW', 'LBBBR', 'LBBBW', 'LCAAW', 'LCAHW', 'LCFYW', 'LDTCW', 'LEXXW', 'LFLYW', 'LGHLW', 'LGSTW', 'LGVCW', 'LIFWZ', 'LNZAW', 'LSEAW', 'LTRYW', 'LUNRW', 'LVROW', 'MACAW', 'MAPSW', 'MAQCW', 'MARXR', 'MBTCR', 'MCAAW', 'MCACR', 'MCACW', 'MCAFR', 'MCAGR', 'MDAIW', 'MFICL', 'MITAW', 'MLECW', 'MMVWW', 'MNTSW', 'MOBXW', 'MSAIW', 'MSSAR', 'MSSAW', 'MTEKW', 'MVLAW', 'MVSTW', 'NBSTW', 'NCACW', 'NCNCW', 'NCPLW', 'NEOVW', 'NETDW', 'NIOBW', 'NKGNW', 'NNAGR', 'NNAGW', 'NNAVW', 'NOVVR', 'NOVVW', 'NPABW', 'NRACW', 'NRSNW', 'NRXPW', 'NUBIW', 'NVACR', 'NVACW', 'NVNIW', 'NVVEW', 'NWTNW', 'NXGLW', 'NXLIW', 'NXPLW', 'OABIW', 'OAKUW', 'OCAXW', 'OCEAW', 'OCSAW', 'ODVWZ', 'ONFOW', 'ONMDW', 'ONYXW', 'OPTXW', 'ORGNW', 'OXBRW', 'OXUSW', 'PAVMZ', 'PAYOW', 'PBAXW', 'PCTTW', 'PEGRW', 'PEPLW', 'PETVW', 'PETWW', 'PFTAW', 'PIIIW', 'PLAOW', 'PLMIW', 'PLTNR', 'PLTNW', 'PMGMW', 'PPHPR', 'PPHPW', 'PPYAW', 'PRENW', 'PRLHW', 'PROCW', 'PRSTW', 'PTIXW', 'PTWOW', 'PUCKW', 'PWUPW', 'PXSAW', 'QDROW', 'QETAR', 'QOMOR', 'QOMOW', 'QSIAW', 'RACYW', 'RCACW', 'RCKTW', 'RCRTW', 'RDZNW', 'RELIW', 'REVBW', 'RFACR', 'RFACW', 'RGTIW', 'RMCOW', 'RMGCW', 'ROCLW', 'RUMBW', 'RVMDW', 'RVPHW', 'RVSNW', 'RWODR', 'RWODW', 'SABSW', 'SAGAR', 'SAITW', 'SATLW', 'SBFMW', 'SCLXW', 'SCRMW', 'SDAWW', 'SEPAW', 'SHFSW', 'SHOTW', 'SHPWW', 'SLACW', 'SLAMW', 'SLDPW', 'SLNAW', 'SMXWW', 'SNAXW', 'SONDW', 'SOUNW', 'SPECW', 'SPKLW', 'SQFTW', 'SRZNW', 'STIXW', 'STRCW', 'STSSW', 'SURGW', 'SVIIR', 'SVIIW', 'SVMHW', 'SVREW', 'SWAGW', 'SWSSW', 'SWVLW', 'SXTPW', 'SYTAW', 'SZZLW', 'TALKW', 'TBLAW', 'TCBPW', 'TENKR', 'TETEW', 'TGAAW', 'THCPW', 'THWWW', 'TLGYW', 'TMTCR', 'TNONW', 'TOIIW', 'TRONW', 'TWLVW', 'UHGWW', 'UKOMW', 'USCTW', 'USGOW', 'VERBW', 'VGASW', 'VIEWW', 'VMCAW', 'VRMEW', 'VSACW', 'VSTEW', 'VWEWW', 'WAVSW', 'WESTW', 'WGSWW', 'WINVR', 'WINVW', 'WKSPW', 'WLDSW', 'WTMAR', 'XBPEW', 'XFINW', 'XOSWW', 'XPDBW', 'YOTAR', 'YOTAW', 'YSBPW', 'ZAPPW', 'ZAZZT', 'ZBZZT', 'ZCARW', 'ZCZZT', 'ZJZZT', 'ZLSWW', 'ZURAW', 'ZVZZT', 'ZWZZT', 'ZXYZ.A', 'ZXZZT']


    tickers = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500 = tickers['Symbol'].values.tolist()
    # tick_list = list(set(si.tickers_nasdaq()).difference(set(invalid_tickers)))
    test = DataCreator(tickers_list=sp500, start_date='2020-09-28')

    test.create_csv("temp2/")
    # print(test.stock_to_pd())
    # print(test.get_yearly_n_info())
    print(test.not_available)
    print(len(test.not_available))
