"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders-short.csv", start_val = 1000000):

    # read the order file
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders_df.index = [time.date() for time in orders_df.index]
    print orders_df.index.values[0]
    secret_date = 2011-06-15
    pd.to_datetime(secret_date, infer_datetime_format=True)
    orders_df[orders_df.index.values != secret_date]
    print orders_df

    # get start and end date
    start_date = orders_df.index.values[0]
    end_date = orders_df.index.values[-1]

    # get symbols
    syms = []
    symbols = orders_df['Symbol'].drop_duplicates()
    for val in symbols.iteritems():
        syms.append(val[1])
    # print syms

    # get values for syms and add cash column
    prices = get_data(syms, pd.date_range(start_date, end_date))
    prices['CASH'] = 1
    prices = prices.drop('SPY',1)
    print
    print prices

    # create trade table
    trades = prices.copy()
    trades.loc[:,:] = 0
    # index = orders_df.index.values
    for index, row in orders_df.iterrows():
        temp = row['Shares'] * ((row['Order'] == 'BUY') * 2 - 1)
        trades.loc[index, row['Symbol']] = trades.loc[index, row['Symbol']] + row['Shares'] * ((row['Order'] == 'BUY') * 2 - 1)
        trades.loc[index, 'CASH'] = trades.loc[index, 'CASH'] + prices.loc[index, row['Symbol']] * temp * -1
        print trades.loc[index,'CASH']
    # print
    # print trades

    # create holding table
    trades.loc[start_date, 'CASH'] = trades.loc[start_date, 'CASH'] + start_val
    temp = trades.iloc[0].copy()
    temp.loc[:] = 0
    # print temp
    for index, row in trades.iterrows():
        row = row + temp
        temp = row
        trades.loc[index] = row
    # print
    # print trades

    # create the value table
    cols = [col for col in trades.columns if col not in 'CASH']
    trades[cols] = trades[cols] * prices[cols]
    print trades

    # leverage
    temp = trades.iloc[0].copy()
    temp.loc[:] = 0
    temp.loc['CASH'] = start_val
    for index, row in orders_df.iterrows():
        # if leverage is above 3.0
        temp = trades.loc[index].copy().abs()
        leverage = temp[cols].sum() / trades.loc[index].sum()
        print leverage
        if leverage >= 3:
            one_day = timedelta(days = 1)
            trades.loc[index] = trades.loc[index - one_day]

    print trades

    print trades
    # add total portfolio value
    trades['VALUE'] = trades.sum(axis = 1)
    # print trades

    portvals = trades['VALUE']
    print portvals
    return portvals

def test_code():
    
    # Define input parameters

    of = "./orders/orders-leverage-2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    start_date = portvals.index.values[0]
    end_date = portvals.index.values[-1]

    daily_return = portvals.pct_change(1)
    avg_daily_ret = daily_return.mean()
    std_daily_ret = daily_return.std()

    # Cumulative return
    cum_ret = portvals.iloc[-1] / portvals.iloc[0] - 1

    # Sharpe ratio
    k = np.sqrt(252)
    sharpe_ratio = avg_daily_ret / std_daily_ret * k

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    # print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    # print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    # print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    # print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
