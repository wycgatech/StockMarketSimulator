"""
Test a learner.
"""

import numpy as np
import math
import sys
import LinRegLearner as lrl
import RTLearner as rt
import BagLearner as bl
import warnings
from random import randint
import time
warnings.simplefilter('ignore')

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    start_time = time.time()
    # compute how much of the data is training and testing
    train_rows = math.floor(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows
    train_start = randint(0, math.floor(0.3 * data.shape[0]))

    # separate out training and testing data
    Xtrain = data[:train_rows,0:-1]
    Ytrain = data[:train_rows,-1]
    Xtest = data[train_rows:,0:-1]
    Ytest = data[train_rows:,-1]


    # # separate out training and testing data
    # Xtrain = data[train_start:train_rows + train_start,0:-1]
    # Ytrain = data[train_start:train_rows + train_start,-1]
    #
    # Xtest = data[0: train_start - 1, 0:-1]
    # print Xtest.shape
    # Xtest2 = data[train_start + train_rows + 1:, 0: -1]
    # print Xtest2.shape
    # Xtest = np.concatenate((Xtest, Xtest2), axis = 0)
    #
    # Ytest = data[0: train_start - 1,-1]
    # Ytest2 = data[train_start + train_rows + 1:,-1]
    # Ytest = np.concatenate((Ytest, Ytest2), axis = 0)

    print Xtest.shape
    print Ytest.shape

    # # create a linear learner and train it
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    # learner.addEvidence(Xtrain, Ytrain) # train it

    # # create a random tree learner and train it
    # learner = rt.RTLearner(leaf_size = 90, verbose = True) # create a LinRegLearner
    # learner.addEvidence(Xtrain, Ytrain) # train it

    # create a bagging learner and train it
    learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 200}, bags=1, boost=False, verbose=False)
    learner.addEvidence(Xtrain, Ytrain)

    # evaluate in sample
    predY = learner.query(Xtrain) # get the predictions
    rmse = math.sqrt(((Ytrain - predY) ** 2).sum()/Ytrain.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytrain)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(Xtest) # get the predictions
    np.savetxt("res_test.csv", predY, delimiter=",")
    predY[predY < 1.9] = 1
    predY[predY >= 2.1] = 3
    predY[np.logical_and(predY >= 1.9, predY < 2.1)] = 2

    rmse = math.sqrt(((Ytest - predY) ** 2).sum()/Ytest.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytest)
    print "corr: ", c[0,1]

    s = np.sum(predY == Ytest)
    print s
    print(float(s)/float(Ytest.shape[0]))

    end_time = time.time()
    duration = end_time - start_time
    print duration




