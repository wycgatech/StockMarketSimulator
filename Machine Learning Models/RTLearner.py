"""
Random Tree Learner(final).  (c) 2016 Yichuan Wang
"""

import numpy as np
import random as rdm

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self._leaf_size = leaf_size
        pass  # move along, these aren't the drones you're looking for

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        newdataX = np.ones([dataX.shape[0], dataX.shape[1] + 1])
        newdataX[:, 0:dataX.shape[1]] = dataX
        newdataX[:, dataX.shape[1]] = dataY
        data = newdataX

        # build and save the model
        def randomTreeBuilder(data):

            if data.shape[0] <= self._leaf_size:
                mean = np.mean(data[:, data.shape[1] - 1])
                return np.array([[-1, mean, -1, -1]])
            else:
                identical = (data[0, data.shape[1] - 1] == data[:, data.shape[1] - 1])
                if np.all(identical == True):
                    return np.array([[-1, data[0,data.shape[1] - 1], -1, -1]])
                else:
                    # print "current input", data
                    i = rdm.randint(0, data.shape[1] - 2)
                    r1 = rdm.randint(0, data.shape[0] - 1)
                    r2 = rdm.randint(0, data.shape[0] - 1)
                    while(data[r1, i] == data[r2, i]):
                        r2 = rdm.randint(0, data.shape[0] - 1)
                        i = rdm.randint(0, data.shape[1] - 2)
                    # print "random number are:", r1, " ", r2

                    splitVal = (data[r1, i] + data[r2, i]) / 2
                    # print "current split value:", splitVal

                    leftTree = randomTreeBuilder(data[data[:,i] <= splitVal])
                    # print "left tree:", leftTree
                    rightTree = randomTreeBuilder(data[data[:,i] > splitVal])
                    # print  "right tree:", rightTree
                    root = np.array([[i, splitVal, 1, leftTree.shape[0] + 1]])
                    # print "current root", root

                    result = np.concatenate((root, leftTree))
                    result = np.concatenate((result, rightTree))

                    return result

        self.tree = randomTreeBuilder(newdataX)
        # print self.tree

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        testData = np.ones([points.shape[0], points.shape[1] + 1])
        testData[:, 0:points.shape[1]] = points
        # print points

        queryResult = []

        for i in xrange(testData.shape[0]):
            current = testData[i, :]
            index = 0
            curNode = self.tree[index, :]
            isLeaf = curNode[0]

            while isLeaf != -1:
                metric = curNode[0]
                if current[metric] <= curNode[1]:
                    index = index + curNode[2]
                    curNode = self.tree[index, :]
                    # print "current node is:", curNode
                    isLeaf = curNode[0]
                else:
                    index = index + curNode[3]
                    curNode = self.tree[index, :]
                    # print "current node is:", curNode
                    isLeaf = curNode[0]
            queryResult.append(curNode[1])

        # print queryResult
        return queryResult


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
