import numpy as np

"""
Turns vector into a diagonal matrix.
This is used especially for eigenvalue/vector calculations.
"""
def as_singular(sigma, rank=None):
    sigma_len = len(sigma)
    if rank is None:
        rank = sigma_len
    array = np.zeros((sigma_len, sigma_len))
    for i in range(rank):
        array[i][i] = sigma[i] if sigma[i] >= 0 else 0
    return array


def eigvector(A, output="all", k=1):
    e, U = np.linalg.eigh(A)
    if output == "all":
        return U
    elif output == "max":
        return U[:,np.argsort(e)[:k]]
    elif output == "min":
        return U[:,np.argsort(-e)[:k]]

def generalized_eigvector(A, B, output="all", k=1):
    # Calculate EVD of B
    e, U = np.linalg.eigh(B)
    E = as_singular(e ** -0.5)
    C = E @ U.T @ A @ U @ E
    
    # Calculating EVD of C
    return eigvector(C, output, k)