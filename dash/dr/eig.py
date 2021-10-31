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


def eig(A, output="all", k=1):
    e, U = np.linalg.eigh(A)
    if output == "all":
        return e, U
    elif output == "max":
        new_e = np.argsort(e)
        return new_e[:k], U[:,new_e[:k]]
    elif output == "min":
        new_e = np.argsort(-e)
        return new_e[:k], U[:,new_e[:k]]

def generalized_eig(A, B, output="all", k=1):
    # Calculate EVD of B
    e, U = np.linalg.eigh(B)
    E = as_singular(e ** -0.5)
    C = E @ U.T @ A @ U @ E
    
    # Calculating EVD of C
    return eig(C, output, k)