"""
Bag Learner(final).  (c) 2016 Yichuan Wang
"""

import numpy as np
import random as rdm
import RTLearner as rt

class BagLearner(object):
    def __init__(self, learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False):
        self.learners = []
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))


    def addEvidence(self, dataX, dataY):

        for l in self.learners:
            l.addEvidence(dataX, dataY)
            # print l.tree
            # print

    def query(self, data):
        # data = np.array(points)
        self.queryResult = []

        for l in self.learners:
            self.queryResult.append(l.query(data))

        toArray = np.array(self.queryResult)
        # print toArray
        mean = toArray.mean(axis=0)

        return mean

