import numpy as np

"""
# Mixture scatter

Calculated by ignoring the groups or labels in the dataset
Mixture scatter (M) = Within class scatter (W) + Between class scatter (B)
"""
def mixture_scatter(y):
    R = np.zeros((y.shape[1], y.shape[1]))
    m = y.shape[0]
    mu = np.mean(y, axis=0)
    for i in range(m):
        R += (y[i] - mu).reshape(-1, 1) @ (y[i] - mu).T.reshape(1, -1)
    return R

"""
Within scatter
"""
def within_scatter(y, g):
    W = np.zeros((y.shape[1], y.shape[1]))
    m = y.shape[0]
    groups = np.unique(g)
    k = len(groups)
    mus = np.zeros((k, y.shape[1]))
    counts = np.zeros(k)
    for i in range(m):
        mus[g[i]] += y[i]
        counts[g[i]] += 1
    for i in range(len(mus)):
        mus[i] /= counts[i]
    for i in range(m):
        W += (y[i] - mus[g[i]]).reshape(-1, 1) @ (y[i] - mus[g[i]]).T.reshape(1, -1)
    return W

"""
Between class scatter
"""
def between_scatter(y, g):
    # Initialization of the between class scatter
    B = np.zeros((y.shape[1], y.shape[1]))
    
    # Other initializations
    m = y.shape[0]          # Number of training examples
    k = len(np.unique(g))   # Number of groups
    
    # Initializations needed to calculate mu(j) for each group j
    mus = np.zeros((k, y.shape[1]))
    counts = np.zeros(k)
    
    # Calculating mu(j)
    for i in range(m):
        mus[g[i]] += y[i]
        counts[g[i]] += 1
    for i in range(len(mus)):
        mus[i] /= counts[i]
        
    # Calculating mu
    mu = sum([mus[i] * counts[i] for i in range(k)]) / sum(counts[i])
        
    # Use formula to calculate between class scatter matrix
    # B = mj(uj - u)(uj - u).T for j in range(k)
    for j in range(k):
        B += counts[i] * (mu[j] - mu).reshape(-1, 1) @ (mu[j] - mu).T.reshape(1, -1)
    return B