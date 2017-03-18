"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import sys
import LinRegLearner as lrl
import RTLearner as rt
import BagLearner as bl
import warnings
import pandas as pd
import datetime as dt
from util import get_close, plot_data, get_high, get_low, get_volume, get_adjclose, get_data
from operator import itemgetter
warnings.simplefilter('ignore')

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    # compute how much of the data is training and testing
    print data.shape
    train_rows = math.floor(1 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    Xtrain = data[:1008,0:-1]
    Ytrain = data[:1008,-1]
    Xtest = data[:1008:,0:-1]
    Ytest = data[:1008:,-1]

    print Xtrain.shape
    print Ytrain.shape

    # THIS KILLS ME!!!
    holidays = [dt.datetime(2006, 01, 02), dt.datetime(2006, 01, 16), dt.datetime(2006, 02, 20),
                dt.datetime(2006, 04, 14), dt.datetime(2006, 05, 29), dt.datetime(2006, 07, 04),
                dt.datetime(2006, 9, 04), dt.datetime(2006, 11, 23), dt.datetime(2006, 12, 25),
                dt.datetime(2007, 01, 01), dt.datetime(2007, 01, 02), dt.datetime(2007, 01, 15), dt.datetime(2007, 02, 19),
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

    # # create a linear learner and train it
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    # learner.addEvidence(Xtrain, Ytrain) # train it

    # create a random tree learner and train it
    learner = rt.RTLearner(leaf_size = 10, verbose = True) # create a LinRegLearner
    learner.addEvidence(Xtrain, Ytrain) # train it

    # # create a bagging learner and train it
    # learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 50}, bags=10, boost=False, verbose=False)
    # learner.addEvidence(Xtrain, Ytrain)

    # evaluate in sample
    predY = learner.query(Xtrain) # get the predictions
    sd = dt.datetime(2006, 01, 01)
    ed = dt.datetime(2009, 12, 31)
    dates = pd.date_range(sd, ed)
    symbols = ['IBM']
    prices = get_data(symbols, dates)[symbols]
    prices = prices.ix[14:-10, :]

    predY = learner.query(Xtrain)
    rmse = math.sqrt(((Ytrain - predY) ** 2).sum()/Ytrain.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytrain)
    print "corr: ", c[0, 1]

    prices_t = prices.ix[:1008, :].copy()

    temp = pd.DataFrame({'Order': predY}, index=prices_t.index)
    prices['Order'] = temp['Order']
    prices = prices[prices.Order != 0]
    print prices_t.shape

    # generate order book
    order_list = []
    next_date = dt.datetime(1992, 06, 29)
    for day in prices.index:
        i = 0
        if day > next_date:
            if prices.ix[day, 'Order'] > 0:
                order_list.append([day.date(), 'IBM', 'BUY', 500])
            elif prices.ix[day, 'Order'] < 0:
                order_list.append([day.date(), 'IBM', 'SELL', 500])

            # get rid of all the actions in between
            current_date = day
            while (i < 10):
                current_date = current_date + dt.timedelta(days=1)
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
            if prices.ix[day, 'Order'] > 0:
                order_list.append([current_date.date(), 'IBM', 'SELL', 500])
            elif prices.ix[day, 'Order'] < 0:
                order_list.append([current_date.date(), 'IBM', 'BUY', 500])
            order_list.sort(key=itemgetter(0))
            next_date = current_date

    text_file = open("RT_orderbook.csv", "w")
    text_file.write("Date,Symbol,Order,Shares\n")
    for order in order_list:
        print ", ".join(str(x) for x in order)
        text_file.write(",".join(str(x) for x in order))
        text_file.write("\n")

    text_file.close()

    # # evaluate out of sample
    # predY = learner.query(Xtest) # get the predictions
    # rmse = math.sqrt(((Ytest - predY) ** 2).sum()/Ytest.shape[0])
    # print
    # print "Out of sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=Ytest)
    # print "corr: ", c[0,1]







