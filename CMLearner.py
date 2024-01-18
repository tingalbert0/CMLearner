# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:30:38 2024

@author: Albert Ting
"""

import numpy as np
import matplotlib.pyplot as plt


class CMLearner:
    def __init__(self, s_dim, o_dim, a_dim, env):
        self.env = env
        self.s_dim = s_dim
        self.o_dim = o_dim # vector of observations size
        self.a_dim = a_dim # vector of action size
        
        self.Q = np.random.normal(0, 0.1, size=(self.s_dim, self.o_dim))
        self.V = np.random.normal(0, 1, size=(self.s_dim, self.a_dim))
        
        self.eta_v = 0.004
        self.eta_q = 0.01
    
    def learn(self, observation, action):
        
        s = np.matmul(self.Q, observation[:, np.newaxis])
        va = np.matmul(self.V, action[:, np.newaxis])
        s_hat = s+va
        
        self.V += self.eta_v*(s - s_hat)*action.T
        self.Q += self.eta_q*(s_hat - s) * observation.T
        
        # normalize columns of V
        column_sums = np.sqrt(np.sum(self.V**2, axis=0))
        self.V = self.V / column_sums

    def act(self, observation, o_star):
        u = np.zeros((self.a_dim, 1))
        
        for i, act in enumerate(range(self.a_dim)):
            a = np.zeros((self.a_dim,1))
            a[act] = 1
            u[i] = (np.dot(self.V, a)).T.dot(self.Q.dot(o_star)-self.Q.dot(observation))
        
        g = np.ones(u.shape)
        # action = np.argmax(u * g)
        action = u * g
        
        return action.flatten()
    
    
    
    
    
    
    
    
    