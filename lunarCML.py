# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:10:56 2024

@author: Albert Ting
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class CMLearner:
    def __init__(self, s_dim, env):
        self.s_dim = s_dim
        self.o_dim = env.observation_space.shape[0] # vector of observations size
        self.a_dim = env.action_space.n # vector of action size
        self.env = env
        
        self.Q = np.random.normal(0, 0.1, size=(m, o_dim))
        self.V = np.random.normal(0, 1, size=(m,a_dim))
        
        self.eta_v = 0.016
        self.eta_q = 0.37
    
    def learn(self, observation, action):
        a = np.zeros((a_dim,1))
        a[action] = 1
        
        
        s = np.matmul(Q, observation[:, np.newaxis])
        va = np.matmul(V, a)
        s_hat = s+va
        self.V += eta_v*(s - s_hat)*a.T
        self.Q += eta_q*(s_hat - s) * observation.T
        
        # normalize columns of V
        column_sums = np.sqrt(np.sum(V**2, axis=0))
        self.V = self.V / column_sums

    def act(self, observation, o_star):
        # print(o_star)
        # print(obseravtion)
        # print(Q.dot(o_star) - Q.dot(observation))
        # print('---')
        u = np.zeros((self.a_dim, 1))
        
        for i, act in enumerate(range(env.action_space.n)):
            a = np.zeros((4,1))
            a[act] = 1
            u[i] = (np.dot(self.V, a)).T.dot(self.Q.dot(o_star)-self.Q.dot(observation))
        g = np.ones(u.shape)
        g[2] = 10
        action = np.argmax(u * g)
        
        return action


# env = gym.make('CartPole-v1', render_mode='human')


env = gym.make('LunarLander-v2')


m = 100 # number of dimensions in high-D space
o_dim = env.observation_space.shape[0] # vector of observations size
a_dim = env.action_space.n # vector of action size


Q = np.random.normal(0, 0.1, size=(m, o_dim))
V = np.random.normal(0, 1, size=(m,a_dim))

eta_v = 0.016
eta_q = 0.37

observation, info = env.reset()

Varray = []
Varray.append(V.flatten())

for _ in range(10000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    a = np.zeros((a_dim,1))
    a[action] = 1
    
    
    s = np.matmul(Q, observation[:, np.newaxis])
    va = np.matmul(V, a)
    s_hat = s+va
    V += eta_v*(s - s_hat)*a.T
    Q += eta_q*(s_hat - s) * observation.T
    
    # normalize columns of V
    column_sums = np.sqrt(np.sum(V**2, axis=0))
    V = V / column_sums
    
    # normalize columns of Q
    column_sums = np.sqrt(np.sum(Q**2, axis=0))
    Q = Q / column_sums
    
    Varray.append(V.flatten())
    
    
    if terminated or truncated:
        observation, info = env.reset()
        
env.close()

plt.plot(Varray)

env = gym.make('LunarLander-v2', render_mode='human')
# env = gym.make('LunarLander-v2')


action = 0
observation, info = env.reset()
for _ in range(1000):
    u = np.zeros((a_dim, 1))
    
    o_star = np.copy(observation)
    # o_star = np.zeros(observation.shape)
    o_star[-1] = 1
    o_star[-2] = 1
    
    # print(o_star)
    # print(obseravtion)
    # print(Q.dot(o_star) - Q.dot(observation))
    # print('---')
    
    for i, act in enumerate(range(env.action_space.n)):
        a = np.zeros((4,1))
        a[act] = 1
        print('Va')
        print(np.dot(V, a))
        print('Qo-Qo')
        print(Q.dot(o_star)-Q.dot(observation))
        u[i] = np.abs((np.dot(V, a)).T.dot(Q.dot(o_star)-Q.dot(observation)))
    
    g = np.ones(u.shape)
    # g[2] = 10
    print(u.T)
    action = np.argmax(u * g)
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    
    
    
    if terminated or truncated:
        observation, info = env.reset()
        
        
env.close()