"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
Yichuan Wang
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import numpy as np
import warnings
import datetime
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        syms = [symbol]
        self.learner = ql.QLearner(100, 3, 0.2, 0.9, 0.99, 0.999, 200, False)

        # prices for calculating indicators
        sd1 = sd - datetime.timedelta(days=30)
        ed1 = ed
        dates1 = pd.date_range(sd1, ed1)
        prices_whole = ut.get_data(syms, dates1)
        prices_benchmark = prices_whole[syms]

        prices = prices_benchmark[sd:ed]
        dates = pd.date_range(sd, ed)
        if self.verbose: print prices

        # SMA
        sma = pd.rolling_mean(prices_benchmark, window=14, min_periods=14)
        # sma_ratio = prices_benchmark / sma

        # Bollinger Band
        rolling_std = pd.rolling_std(prices_benchmark, window=14, min_periods=14)
        top_band = sma + (2 * rolling_std)
        bottom_band = sma - (2 * rolling_std)
        bbp = (prices_benchmark - bottom_band) / (top_band - bottom_band) * 100

        # RSI
        daily_rets = prices_benchmark.copy()
        daily_rets.values[1:, :] = prices_benchmark.values[1:, :] - prices_benchmark.values[:-1, :]
        daily_rets.values[0, :] = np.nan

        up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
        down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()

        up_gain = prices_benchmark.copy()
        up_gain.ix[:, :] = 0
        up_gain.values[14:, :] = up_rets.values[14:, :] - up_rets.values[:-14, :]
        down_loss = prices_benchmark.copy()
        down_loss.ix[:, :] = 0
        down_loss.values[14:, :] = down_rets.values[14:, :] - down_rets.values[:-14, :]

        rs = (up_gain / 14) / (down_loss / 14)
        rsi = 100 - (100 / (1 + rs))
        rsi.ix[:14, :] = np.nan

        rsi[rsi == np.inf] = 100

        # Discretized
        bins = np.linspace(0, 100, 10)
        bbp_bins = np.digitize(bbp, bins) - 1
        rsi_bins = np.digitize(rsi, bins) - 1
        bbp_bins[bbp_bins < 0] = 0
        bbp_bins[bbp_bins > 9] = 9
        rsi_bins[rsi_bins < 0] = 0
        rsi_bins[rsi_bins > 9] = 9

        indicators = prices_benchmark.copy()
        indicators['BBP'] = bbp_bins
        indicators['RSI'] = rsi_bins

        is_converge = False
        num_iter = 0
        trade_days = prices.index.values

        while not is_converge:
            # calculate first indicator value x
            x = indicators.loc[trade_days[0], 'BBP'] * 10 + indicators.loc[trade_days[0], 'RSI']
            action = self.learner.querysetstate(x)
            price_pre = indicators.loc[trade_days[0], symbol]
            position = 0

            for day in trade_days[1:]:
                # action = 0: exit; 1: go long; 2: go short
                #
                if position == -500:
                    if action == 0:
                        position = 0
                    elif action == 1:
                        position = 500
                    else:
                        position = -500
                elif position == 500:
                    if action == 0:
                        position = 0
                    elif action == 2:
                        position = -500
                    else:
                        position = 500
                else:
                    if action == 1:
                        position = 500
                    elif action == 2:
                        position = -500
                    else:
                        position = 0

                # calculate rewards
                reward = (prices.loc[day, symbol] - price_pre) * position
                price_pre = prices.loc[day, symbol]

                x = indicators.loc[day, 'BBP'] * 10 + indicators.loc[day, 'RSI']
                action = self.learner.query(x, reward)

            num_iter += 1
            # check converge is true of not
            if num_iter > 10:
                is_converge = True

        # example use with new colname
        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[syms]
        trades = prices_all[[symbol, ]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later

        # SMA
        sma = pd.rolling_mean(prices, window=14, min_periods=14)

        # Bollinger Band
        rolling_std = pd.rolling_std(prices, window=14, min_periods=14)
        top_band = sma + (2 * rolling_std)
        bottom_band = sma - (2 * rolling_std)
        bbp = (prices - bottom_band) / (top_band - bottom_band) * 100

        # RSI
        daily_rets = prices.copy()
        daily_rets.values[1:, :] = prices.values[1:, :] - prices.values[:-1, :]
        daily_rets.values[0, :] = np.nan

        up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
        down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()

        up_gain = prices.copy()
        up_gain.ix[:, :] = 0
        up_gain.values[14:, :] = up_rets.values[14:, :] - up_rets.values[:-14, :]
        down_loss = prices.copy()
        down_loss.ix[:, :] = 0
        down_loss.values[14:, :] = down_rets.values[14:, :] - down_rets.values[:-14, :]

        rs = (up_gain / 14) / (down_loss / 14)
        rsi = 100 - (100 / (1 + rs))
        rsi.ix[:14, :] = np.nan

        rsi[rsi == np.inf] = 100

        # Discretized
        bins = np.linspace(0, 100, 10)
        bbp_bins = np.digitize(bbp, bins) - 1
        rsi_bins = np.digitize(rsi, bins) - 1
        bbp_bins[bbp_bins < 0] = 0
        bbp_bins[bbp_bins > 9] = 9
        rsi_bins[rsi_bins < 0] = 0
        rsi_bins[rsi_bins > 9] = 9

        indicators = prices.copy()
        indicators['BBP'] = bbp_bins
        indicators['RSI'] = rsi_bins

        trade_days = indicators.index.values

        # here we build a fake set of trades
        # your code should return the same sort of data

        trades.values[:,:] = 0 # set them all to nothing
        # trades.values[3,:] = 500 # add a BUY at the 4th date
        # trades.values[5,:] = -500 # add a SELL at the 6th date
        # trades.values[6,:] = -500 # add a SELL at the 7th date
        # trades.values[8,:] = 1000 # add a BUY at the 9th date

        # if self.verbose: print type(trades) # it better be a DataFrame!
        # if self.verbose: print trades
        # if self.verbose: print prices_all

        x = indicators.loc[trade_days[0], 'BBP'] * 10 + indicators.loc[trade_days[0], 'RSI']
        action = self.learner.querysetstate(x)
        position = 0

        for day in trade_days[1:]:
            # date = pd.to_datetime(day).date()
            if position == -500:
                if action == 0:
                    position = 0
                    # trades.values[date:] = 500
                    trades.loc[day, syms] = 500
                elif action == 1:
                    position = 500
                    # trades.values[date:] = 1000
                    trades.loc[day, syms] = 1000
                else:
                    position = -500
            elif position == 500:
                if action == 0:
                    position = 0
                    # trades.values[date:] = -500
                    trades.loc[day, syms] = -500
                elif action == 2:
                    position = -500
                    # trades.values[date:] = -1000
                    trades.loc[day, syms] = -1000
                else:
                    position = 500
            else:
                if action == 1:
                    position = 500
                    # trades.values[date:] = 500
                    trades.loc[day, syms] = 500
                elif action == 2:
                    position = -500
                    # trades.values[date:] = -500
                    trades.loc[day, syms] = -500
                else:
                    position = 0

            x = indicators.loc[day, 'BBP'] * 10 + indicators.loc[day, 'RSI']
            action = self.learner.querysetstate(x)
            # print position

        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy"
