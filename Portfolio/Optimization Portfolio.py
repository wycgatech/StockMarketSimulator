"""Optimize a portfolio. From Yichuan Wang"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
from scipy.optimize import minimize


def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=[], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    initial_guess = 1.0 / len(syms)
    ini_allocs = []
    for s in syms:
        ini_allocs.append(initial_guess)

    bnds = tuple((0,1) for t in ini_allocs)

    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

    def fun(x):

        prices_processed = prices / prices.iloc[0] * x
        prices_processed['Daily Value'] = prices_processed.sum(1)

        # Average and standard deviation of daily return
        daily_return = prices_processed['Daily Value'].pct_change(1)
        adr = daily_return.mean()
        sddr = daily_return.std()

        # Sharpe ratio
        k = np.sqrt(252)
        modified_sr = adr / sddr * k * (-1)

        return modified_sr

    result_allocs = minimize(fun, ini_allocs, method='SLSQP', bounds=bnds, constraints=cons).x

    prices_processed = prices / prices.iloc[0] * result_allocs
    prices_processed['Daily Value'] = prices_processed.sum(1)

    prices_SPY['Daily Value'] = prices_SPY / prices_SPY.iloc[0]

    # Get daily portfolio value
    port_val = prices_SPY['Daily Value']
    cr, adr, sddr, sr = stats(prices_processed)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # plot
        df_temp = pd.concat([prices_processed['Daily Value'], prices_SPY['Daily Value']], keys=['Portfolio', 'SPY'], axis=1)
        output_plot = df_temp.plot(title = 'Daily Portfolio Value and SPY')
        output_plot.set_xlabel('Date')
        output_plot.set_ylabel('Price')
        pass

    return result_allocs, cr, adr, sddr, sr

def stats(prices_processed):

    # Average and standard deviation of daily return
    daily_return = prices_processed['Daily Value'].pct_change(1)
    adr = daily_return.mean()
    sddr = daily_return.std()

    # Cumulative return
    cr = prices_processed['Daily Value'].iloc[-1] / prices_processed['Daily Value'].iloc[0] - 1

    # Sharpe ratio
    k = np.sqrt(252)
    sr = adr / sddr * k

    return cr, adr, sddr, sr

def test_code():

    start_date = dt.datetime(2010,01,01)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    test_code()
