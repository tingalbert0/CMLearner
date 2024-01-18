# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:41:32 2024

@author: Albert Ting
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from CMLearner import CMLearner



env = gym.make('Ant-v4')


s_dim = 120 # number of dimensions in high-D space
o_dim = env.observation_space.shape[0]
a_dim = 2**env.action_space.shape[0]
CML = CMLearner(s_dim, o_dim, a_dim, env)

observation, info = env.reset()


for _ in range(100000):
    
    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    
    CML.learn(observation, action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()


env = gym.make('Ant-v4', render_mode='human')
observation, info = env.reset()
o_star = observation
o_star[1] = 10
o_star[2] = 10

for _ in range(10000):
    
    action = CML.act(observation, o_star)
    observation, reward, terminated, truncated, info = env.step(action)
    
    CML.learn(observation, action)
    
    # if terminated or truncated:
    #     observation, info = env.reset()

env.close()
















