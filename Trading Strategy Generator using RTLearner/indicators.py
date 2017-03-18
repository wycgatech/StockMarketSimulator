"""ADL indicator. From Yichuan Wang"""

from operator import itemgetter
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from cycler import cycler
from util import get_close, plot_data, get_high, get_low, get_volume, get_adjclose


def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=[], gen_plot=True):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)

    prices = get_close(syms,dates)[syms]
    prices_high = get_high(syms, dates)[syms]
    prices_low = get_low(syms, dates)[syms]
    prices_adjclose = get_adjclose(syms, dates)[syms]

    prices['Close'] = prices['IBM']
    prices = prices.drop('IBM', 1)
    prices['High'] = prices_high['IBM']
    prices['Low'] = prices_low['IBM']
    prices['Adj Close'] = prices_adjclose['IBM']
    print prices_adjclose.shape
    volume = get_volume(syms, dates)[syms]

    # THIS KILLS ME!!!
    holidays = [dt.datetime(2006, 01, 02), dt.datetime(2006, 01, 16), dt.datetime(2006, 02, 20),
                dt.datetime(2006, 04, 14), dt.datetime(2006, 05, 29), dt.datetime(2006, 07, 04),
                dt.datetime(2006, 9, 04), dt.datetime(2006, 11, 23), dt.datetime(2006, 12, 25),
                dt.datetime(2007, 01, 01), dt.datetime(2007, 01, 02), dt.datetime(2007, 01, 15),
                dt.datetime(2007, 02, 19),
                dt.datetime(2007, 04, 06), dt.datetime(2007, 05, 28), dt.datetime(2007, 07, 04),
                dt.datetime(2007, 9, 03), dt.datetime(2007, 11, 22), dt.datetime(2007, 12, 25),
                dt.datetime(2008, 01, 01), dt.datetime(2008, 01, 21), dt.datetime(2008, 02, 18),
                dt.datetime(2008, 03, 21), dt.datetime(2008, 05, 26), dt.datetime(2008, 07, 04),
                dt.datetime(2008, 9, 01), dt.datetime(2008, 11, 27), dt.datetime(2008, 12, 25),
                dt.datetime(2009, 01, 01), dt.datetime(2009, 01, 19), dt.datetime(2009, 02, 16),
                dt.datetime(2009, 04, 10), dt.datetime(2009, 05, 25), dt.datetime(2009, 07, 03),
                dt.datetime(2009, 9, 07), dt.datetime(2009, 11, 26), dt.datetime(2009, 12, 25),
                dt.datetime(2010, 01, 01), dt.datetime(2010, 01, 18), dt.datetime(2010, 02, 15),
                dt.datetime(2010, 04, 02), dt.datetime(2010, 05, 31), dt.datetime(2010, 07, 05),
                dt.datetime(2010, 9, 06), dt.datetime(2010, 11, 25), dt.datetime(2010, 12, 24),
                ]
    # SMA
    sma = prices_adjclose.rolling(window=14, min_periods=14).mean()
    sma_ratio = prices_adjclose / sma

    # ADL
    prices['MFM'] = ((prices['Close'] - prices['Low']) - (prices['High'] - prices['Close'])) / (prices['High'] - prices['Low'])
    prices['MFV'] = prices['MFM'] * volume['IBM']
    prices['ADL'] = prices['MFV'].cumsum()
    adl = pd.DataFrame({'IBM': prices['ADL']})

    # EMA
    ema = prices_adjclose.ewm(ignore_na=False,span=10,min_periods=0,adjust=True).mean()
    ema_ratio = prices_adjclose / ema

    # MFI
    price_chg = prices_adjclose.copy()
    price_chg.values[1:, :] = prices_adjclose.values[1:, :] - prices_adjclose.values[:-1, :]
    price_chg.values[0, :] = 1
    price_chg[price_chg < 0] = -1
    price_chg[price_chg > 0] = 1
    prices['Change'] = price_chg
    prices['TP'] = (prices['High'] + prices['Low'] + prices['Close']) / 3
    PRMF = prices['TP'] * prices['Change'] * volume['IBM']
    NRMF = PRMF.copy()
    PRMF[PRMF < 0] = 0
    NRMF[NRMF > 0] = 0
    prices['PRMF'] = PRMF
    prices['NRMF'] = NRMF
    prices['MFR'] = prices['PRMF'].rolling(min_periods=14, window=14, center=False).sum() / prices['NRMF'].rolling(min_periods=14, window=14, center=False).sum() * -1
    prices['MFI'] = 100 - 100/(1 + prices['MFR'])
    mfi = pd.DataFrame({'IBM': prices['MFI']})

    # bbp
    rolling_std = prices_adjclose.rolling(window=14, min_periods=14).std()
    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)
    bbp = (prices_adjclose - bottom_band) / (top_band - bottom_band)

    # RSI
    rs = prices_adjclose.copy()
    rsi = prices_adjclose.copy()
    daily_rets = prices_adjclose.copy()
    daily_rets.values[1:, :] = prices_adjclose.values[1:, :] - prices_adjclose.values[:-1, :]
    daily_rets.values[0, :] = np.nan

    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()

    up_gain = prices_adjclose.copy()
    up_gain.ix[:, :] = 0
    up_gain.values[14:, :] = up_rets.values[14:, :] - up_rets.values[:-14, :]
    down_loss = prices_adjclose.copy()
    down_loss.ix[:, :] = 0
    down_loss.values[14:, :] = down_rets.values[14:, :] - down_rets.values[:-14, :]

    rs = (up_gain / 14) / (down_loss / 14)
    rsi = 100 - (100 / (1 + rs))
    rsi.ix[:14, :] = np.nan

    rsi[rsi == np.inf] = 100

    # Combine indicators
    indicators = prices_adjclose.copy()
    # indicators['SMA Ratio'] = sma_ratio
    # indicators['EMA Ratio'] = ema_ratio
    # indicators['ADL'] = adl
    indicators['MFI'] = mfi
    indicators['BBP'] = bbp
    indicators['RSI'] = rsi
    indicators = indicators.drop('IBM', 1)

    # Generate order based on indicators
    orders = prices_adjclose.copy()
    orders.ix[:, :] = np.NaN
    # Create a binary (0-1) array showing when price is above SMA-14.
    sma_cross = pd.DataFrame(0, index=sma_ratio.index, columns=sma_ratio.columns)
    sma_cross[sma_ratio >= 1] = 1
    # Turn that array into one that only shows the crossings (-1 == cross down, +1 == cross up).
    sma_cross[1:] = sma_cross.diff()
    sma_cross.ix[0] = 0
    # 0.46
    orders[(mfi < 29) & (bbp < 0) | (rsi < 30) | (sma_ratio < 0.8)] = 500
    orders[(mfi > 51) & (bbp > 0.6) | (rsi > 51) | (sma_ratio > 1.1)] = -500

    orders[(sma_cross != 0)] = 0
    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)
    orders[1:] = orders.diff()
    orders.ix[0] = 0
    orders = orders.loc[(orders != 0).any(axis=1)]

    # generate order book
    order_list = []
    next_date = dt.datetime(1992,06,29)
    for day in orders.index:
        i = 0
        if day > next_date:
            if orders.ix[day, 'IBM'] > 0:
                order_list.append([day.date(), 'IBM', 'BUY', 500])
            elif orders.ix[day, 'IBM'] < 0:
                order_list.append([day.date(), 'IBM', 'SELL', 500])

            # get rid of all the actions in between
            current_date = day

            while(i < 10):
                current_date = current_date + datetime.timedelta(days = 1)
                weekday = current_date.weekday()
                if weekday < 5:
                    is_holiday = False
                    for h in holidays:
                        if current_date.date() == h.date():
                            is_holiday = True
                            break
                    if is_holiday != True:
                        i = i + 1
            # exit actions
            if orders.ix[day, 'IBM'] > 0:
                order_list.append([current_date.date(), 'IBM', 'SELL', 500])
            elif orders.ix[day, 'IBM'] < 0:
                order_list.append([current_date.date(), 'IBM', 'BUY', 500])
            order_list.sort(key=itemgetter(0))
            next_date = current_date

    text_file = open("rule_based_orderbook.csv", "w")
    text_file.write("Date,Symbol,Order,Shares\n")
    for order in order_list:
        # print ", ".join(str(x) for x in order)
        text_file.write(",".join(str(x) for x in order))
        text_file.write("\n")
    text_file.close()

    # generate X, Y for RT learner
    training = indicators.ix[14:-10].copy()
    temp0 = prices_adjclose.ix[14:-10]
    temp0 = temp0.reset_index(drop=True)
    temp10 = prices_adjclose.ix[24:]
    temp10 = temp10.reset_index(drop=True)
    temp10['IBM'] = temp10['IBM'] - temp0['IBM']
    for day in temp10.index:
        if temp10.ix[day, 'IBM'] > 1:
            temp10.ix[day, 'IBM'] = 1
        elif temp10.ix[day, 'IBM'] < -1:
            temp10.ix[day, 'IBM'] = -1
        else:
            temp10.ix[day, 'IBM'] = 0
    temp = temp10['IBM'].values.tolist()

    result = pd.DataFrame({'Y': temp}, index = training.index)
    training['Y'] = result['Y']
    # normalize
    training['BBP'] = training['BBP'] * 100
    training.to_csv('./RTlearner_input.csv', header=None, index=None, sep=',', mode='a')

    if gen_plot:

        # plot ADL
        # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
        #
        # plt.rc('lines', linewidth=1)
        # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
        #                            cycler('linestyle', ['-', '--', ':', '-.'])))
        #
        # ADL = prices['ADL']
        # ax1.plot(prices['High'], label = "Low price")
        # ax1.plot(prices['Low'], label = "High price")
        # ax1.plot(prices['Close'], label = "Close price")
        # ax1.plot(prices['Adj Close'], label = "Adjusted Close")
        # plt.suptitle('Accumulation Distribution Line', fontsize=20)
        # ax1.set_ylabel('Price')
        # ax1.legend(loc = 'upper left', shadow = True, prop = {'size': 9})
        # ax1.grid()
        #
        # ax2.plot(volume/1000000, label = "Volume", color = "g")
        # ax2.set_ylabel('Volume (million)')
        # ax2.legend(loc='upper left', shadow=True, prop = {'size': 11})
        # ax2.grid()
        #
        # ax3.plot(ADL/1000000, label = "ADL", color = "r")
        # ax3.legend(loc='upper left', shadow=True, prop = {'size': 11})
        # ax3.set_xlabel('Time')
        # ax3.set_ylabel('ADL')
        # ax3.grid()

        # plot SMA
        # fig, (ax1, ax2) = plt.subplots(nrows=2)
        # plt.suptitle('Simple Moving Average (SMA) Ratio)', fontsize = 20)
        # ax1.plot(sma, color = 'b', label = 'SMA')
        # ax1.set_ylabel("SMA")
        # ax1.legend(loc='upper left', shadow=True, prop={'size': 9})
        # ax1.grid()
        # ax3 = ax1.twinx()
        # ax3.plot(prices_adjclose, color = 'r', label = 'Adjusted Close')
        # ax3.set_ylabel("Adjusted Close")
        # ax3.legend(loc='lower right', shadow=True, prop={'size': 9})
        #
        # ax2.plot(sma_ratio, color = 'g', label = 'SMA ratio')
        # ax2.set_xlabel('Time')
        # ax2.set_ylabel('SMA ratio')
        # ax2.grid()
        # ax2.legend(loc='upper left', shadow=True, prop={'size': 9})

        # plot EMA
        # fig, (ax1, ax3) = plt.subplots(nrows = 2)
        # ax1.plot(ema, color = 'r', label = "EMA")
        # ax1.set_xlabel("Time")
        # ax1.set_ylabel("EMA")
        # ax1.grid()
        # ax1.legend(loc = 'upper left', shadow = True, prop = {'size': 12})
        # plt.suptitle("Exponential Moving Average (EMA)", fontsize=20)
        #
        # ax2 = ax1.twinx()
        # ax2.plot(sma, color = 'b', label = 'SMA')
        # ax2.set_ylabel('SMA')
        # ax2.legend(loc='lower right', shadow = True, prop={'size': 12})
        #
        # ax3.plot(prices_adjclose, color = 'g', label = 'Adjusted Close')
        # ax3.set_ylabel('Price')
        # ax3.legend(loc = 'upper left', shadow = True, prop = {'size': 12})
        # ax3.grid()

        # plot MFI
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)

        plt.rc('lines', linewidth=1)
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
                                   cycler('linestyle', ['-', '--', ':', '-.'])))

        ADL = prices['ADL']
        ax1.plot(prices['High'], label = "Low price")
        ax1.plot(prices['Low'], label = "High price")
        ax1.plot(prices['Close'], label = "Close price")
        ax1.plot(prices['Adj Close'], label = "Adjusted Close")
        plt.suptitle('Money Flow Index', fontsize=20)
        ax1.set_ylabel('Price')
        ax1.legend(loc = 'upper left', shadow = True, prop = {'size': 9})
        ax1.grid()

        ax2.plot(volume/1000000, label = "Volume", color = "g")
        ax2.set_ylabel('Volume (million)')
        ax2.legend(loc='upper left', shadow=True, prop = {'size': 11})
        ax2.grid()

        ax3.plot(mfi, label = "MFI", color = "r")
        ax3.legend(loc='upper left', shadow=True, prop = {'size': 11})
        ax3.set_xlabel('Time')
        ax3.set_ylabel('MFI')
        ax3.grid()

        # # Bollinger Band
        # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
        # ax1.plot(sma, label = 'SMA')
        # ax1.plot(prices_adjclose, label = "Adjusted Close")
        # ax1.legend(loc = 'upper left', shadow = True, prop = {'size': 12})
        #
        # ax2.plot(rolling_std, label = "Rolling Standard Deviation")
        # ax2.set_ylabel('STD')
        # ax2.legend(loc = 'upper left', shadow = True, prop = {'size': 12})
        #
        # ax3.plot(bbp, label = "Bollinger Band", color = 'g')
        # ax3.set_ylabel('Bollinger Band')
        # ax3.legend(loc = "upper left", shadow = True, prop = {'size': 10})
        #
        # plt.suptitle('Bollinger Band', fontsize = 20)
        # ax1.grid()
        # ax2.grid()
        # ax3.grid()

        plt.subplots_adjust(hspace=0)
        plt.show()
        pass

def test_code():


    start_date = dt.datetime(2006,01,01)
    end_date = dt.datetime(2009,12,31)
    symbols = ['IBM']

    # Assess the portfolio
    optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=True)

if __name__ == "__main__":

    test_code()
