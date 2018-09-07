# DongKyu Kim
# Assignment Week 1
# ECE471 CGML
# Professor Curro

import numpy as np
import tensorflow as tf

from tqdm import tqdm

# Hyper Parameters
M = 4

# data generation

def data():
    n = 50
    sigma = 0.1
    np.random.seed(31415)
    for _ in range(n):
        x = np.random.uniform(0,1)
        y = np.sin(2*np.pi*x) + sigma * np.random.normal()

        yield x, y

