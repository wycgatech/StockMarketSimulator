"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "output.csv", start_val = 100000):

    # read the order file
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders_df.index = [time.date() for time in orders_df.index]
    secret_date = 2011-06-15
    pd.to_datetime(secret_date, infer_datetime_format=True)
    orders_df[orders_df.index.values != secret_date]
    # print orders_df

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
    prices = prices.drop('SPY', 1)
    print
    # print prices

    # create trade table
    trades = prices.copy()
    trades.loc[:,:] = 0
    # index = orders_df.index.values
    for index, row in orders_df.iterrows():
        temp = row['Shares'] * ((row['Order'] == 'BUY') * 2 - 1)
        trades.loc[index, row['Symbol']] = trades.loc[index, row['Symbol']] + row['Shares'] * ((row['Order'] == 'BUY') * 2 - 1)
        trades.loc[index, 'CASH'] = trades.loc[index, 'CASH'] + prices.loc[index, row['Symbol']] * temp * -1
        # print trades.loc[index,'CASH']
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
    # print trades

    # leverage
    temp = trades.iloc[0].copy()
    temp.loc[:] = 0
    temp.loc['CASH'] = start_val
    for index, row in orders_df.iterrows():
        # if leverage is above 3.0
        temp = trades.loc[index].copy().abs()
        leverage = temp[cols].sum() / trades.loc[index].sum()
        # print leverage
        if leverage >= 3:
            one_day = timedelta(days = 1)
            trades.loc[index] = trades.loc[index - one_day]

    # print trades
    #
    # print trades
    # add total portfolio value
    trades['VALUE'] = trades.sum(axis = 1)
    # print trades
    check = 0
    long = []
    short = []
    exit = []
    for day in orders_df.index:
        if check == 0:
            if orders_df.ix[day, 'Order'] == 'BUY':
                long.append(day)
                check = 1
            elif orders_df.ix[day, 'Order'] == 'SELL':
                short.append(day)
                check = 1
        else:
            exit.append(day)
            check = 0

    portvals = trades['VALUE']
    order_vales = pd.DataFrame({'Rule_base': trades['VALUE']})
    order_vales['Rule_base'] = order_vales['Rule_base'] / 100000
    rem_val = 100000 - prices['IBM'].iloc[0] * 500
    order_vales['Benchmark'] = (prices['IBM'] * 500 + rem_val) / 100000

    # print order_vales

    return order_vales, long, short, exit
    # return portvals

def test_code():
    # Define input parameters

    of = "rule_based_orderbook.csv"
    of2 = "RT_orderbook.csv"
    of3 = "benchmark_orderbook.csv"
    sv = 100000

    # Process orders
    portvals, l, s, e = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Process orders
    portvals2, l2, s2, e2 = compute_portvals(orders_file=of2, start_val=sv)
    if isinstance(portvals2, pd.DataFrame):
        portvals2 = portvals2[portvals2.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Process orders
    portvals3, l3, s3, e3 = compute_portvals(orders_file=of3, start_val=sv)
    if isinstance(portvals3, pd.DataFrame):
        portvals3 = portvals3[portvals3.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # plot
    fig, ax1 = plt.subplots(nrows=1)
    ax1.plot(portvals, label="Manual Rule-based Trader", color='blue')
    ax1.plot(portvals3, label="Benchmark", color='black')
    ax1.plot(portvals2, label="ML Trader", color='green')
    plt.ylim([0.8, 2.5])

    ax1.legend(loc='upper left', shadow=True, prop={'size': 12})
    ax1.vlines(l2, ymin=0.8, ymax=2.5, color='green', linestyle='solid')
    ax1.vlines(s2, ymin=0.8, ymax=2.5, color='red', linestyle='solid')
    ax1.vlines(e2, ymin=0.8, ymax=2.5, color='black', linestyle='solid')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    plt.suptitle("ML trader VS. Manual Rule-Based VS. Benchmark", fontsize=20)

    plt.grid()
    fig.autofmt_xdate()
    plt.show()

    start_date = portvals2.index.values[0]
    end_date = portvals2.index.values[-1]

    daily_return = portvals2.pct_change(1)
    avg_daily_ret = daily_return.mean()
    std_daily_ret = daily_return.std()

    # Cumulative return
    cum_ret = portvals2.iloc[-1] / portvals2.iloc[0] - 1

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
    print "Final Portfolio Value: {}".format(portvals2[-1])

if __name__ == "__main__":
    test_code()
