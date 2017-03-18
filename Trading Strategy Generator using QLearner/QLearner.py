"""
Template for implementing QLearner  (c) 2015 Tucker Balch
Yichuan Wang
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states = 100, \
        num_actions = 3, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.9, \
        radr = 0.99, \
        dyna = 200, \
        verbose = False):

        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.q_table = np.random.uniform(-1, 1, size = num_states * num_actions).reshape((num_states, num_actions))
        self.r_table = []

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if_random = rand.random()
        if if_random < self.rar:
            action = rand.randint(0, 2)
        else:
            action = np.argmax(self.q_table[s])

        self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        if_random = rand.random()
        if if_random < self.rar:
            action = rand.randint(0, 2)
        else:
            action = np.argmax(self.q_table[s_prime])

        update_val = (1 - self.alpha) * self.q_table[self.s][self.a] + self.alpha * (r + self.gamma * self.q_table[s_prime][np.argmax(self.q_table[s_prime])])
        self.q_table[self.s][self.a] = update_val

        self.r_table.append([self.s, self.a, s_prime, r])
        self.rar = self.rar * self.radr
        self.s = s_prime
        self.a = action

        start = rand.randint(0, len(self.r_table) - 1)
        iter = 0
        n = 50
        while iter < n and start < len(self.r_table):
            temp_s = self.r_table[start][0]
            temp_a = self.r_table[start][1]
            temp_sp = self.r_table[start][2]
            temp_r = self.r_table[start][3]
            update_val = (1 - self.alpha) * self.q_table[temp_s][temp_a] + self.alpha * (temp_r + self.gamma * self.q_table[temp_sp][np.argmax(self.q_table[temp_sp])])
            self.q_table[temp_s][temp_a] = update_val
            self.rar = self.rar * self.radr
            iter = iter + 1
            start = start + 1

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
