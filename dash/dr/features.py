import numpy as np
import math
from scipy import linalg

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
    mu = sum([mus[i] * counts[i] for i in range(k)]) / sum(counts)
        
    # Use formula to calculate between class scatter matrix
    # B = mj(uj - u)(uj - u).T for j in range(k)
    for j in range(k):
        B += counts[i] * (mus[j] - mu).reshape(-1, 1) @ (mus[j] - mu).T.reshape(1, -1)
    return B

"""
# Pearson Selection

Calculates Pearson coefficient of all features in the dataset.
Returns an array of all the coefficients for each feature
"""
def pearson_selection(x, y, k=2):
    # Initializations
    mu_x = np.zeros(x.shape[1])
    mu_y = 0.0
    s_x = np.zeros(x.shape[1])
    s_y = 0.0
    c_y = np.zeros(x.shape[1])
    R = np.zeros(x.shape[1])
    
    # Calculate mus
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            mu_x[j] += x[i][j]
        mu_y += y[i]   
    for val in mu_x:
        val /= x.shape[0]
    mu_y /= x.shape[0]
    
    # Calculate scatters
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            s_x[j] += (x[i][j] - mu_x[j]) ** 2
            c_y[j] += (x[i][j] - mu_x[j]) * (y[i] - mu_y)
        s_y += (y[i] - mu_y) ** 2
    
    # Create R matrix with coefficients
    for j in range(x.shape[1]):
        R[j] = abs(c_y[j] / math.sqrt(s_x[j] * s_y))
        
    # Select features with highest coefficients
    return x[:, np.argsort(-R)[:k]]

"""
# Fisher Selection

Between class scatter divided by within class scatter
"""
def fisher_selection(x, y, k=2):
    # Calculate between-class divided by within-class
    R = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        W = within_scatter(x[:,j].reshape(-1, 1), y)[0][0]
        B = between_scatter(x[:,j].reshape(-1, 1), y)[0][0]
        R[j] = B / W
        
    # Select features with highest R
    return x[:, np.argsort(-R)[:k]]
    
def recursive_feat_elimination(x, y, k=2):
    # Base case
    if x.shape[1] <= k:
        return x
    # Otherwise
    B = x.T @ x
    h = x.T @ y
    a = (np.linalg.inv(B) @ h).reshape(-1,)
    i = np.argmin(np.abs(a))
    return recursive_feat_elimination(np.concatenate((x[:,:i], x[:,i+1:]), axis=1), y, k)

def qr_selection(x, k=2):
    _, _, P = linalg.qr(x, pivoting=True)
    return x[:,P[:k]]