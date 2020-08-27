import numpy as np
import time
from tqdm import tqdm
from numba import jit, vectorize, float64, int64, njit, prange
import math
import random
import sys
import getopt
import sympy 

def lp_original(trX, X, Z, sigma_X = None, sigma_A = None):
    N, D = X.shape
    K = Z.shape[1]
    invMat = np.linalg.inv(Z.T @ Z + sigma_X**2 / sigma_A**2 * np.eye(K))
    res = - N * D / 2 * np.log(2 * np.pi)
    res -= (N - K) * D * np.log(sigma_X)
    res -= K * D * np.log(sigma_A)
    res += D / 2 * np.log(np.linalg.det(invMat))
    res -= 1 / (2 * sigma_X**2) * np.trace(X.T @ (np.eye(N) - Z @ invMat @ Z.T) @ X)
    return res   

def lp(trX, X, Z = None, sigma_X = None, sigma_A = None):
    N, D = X.shape
    K = Z.shape[1]
    u, s, _ = np.linalg.svd(Z,full_matrices=False)
    det = np.sum(np.log(s**2 + sigma_X**2 / sigma_A**2))
    l = s**2 / (s**2 + sigma_X**2 / sigma_A**2)
    uTX = u.T @ X
    uX = np.sum(uTX ** 2, axis = 1)

    res = - N * D / 2 * np.log(2 * np.pi)
    res -= (N - K) * D * np.log(sigma_X)
    res -= K * D * np.log(sigma_A)
    res -= D / 2 * det
    res -= 1 / (2 * sigma_X**2) * (trX - sum(l * uX))
    return res

@jit
def matrix_multiply_numba(A, B):
    m, n = A.shape
    n, p = B.shape
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i,j] += A[i,k] * B[k, j]
    return C

@jit(float64(float64, float64[:,:], float64[:,:], float64, float64))
def lp_numba(trX, X, Z = None, sigma_X = None, sigma_A = None):
    N, D = X.shape
    K = Z.shape[1]
    u, s, _ = np.linalg.svd(Z,full_matrices=False)
    det = np.sum(np.log(s**2 + sigma_X**2 / sigma_A**2))
    l = s**2 / (s**2 + sigma_X**2 / sigma_A**2)
    uTX = matrix_multiply_numba(u.T, X)
    uX = np.sum(uTX ** 2, axis = 1)

    res = - N * D / 2 * np.log(2 * np.pi)
    res -= (N - K) * D * np.log(sigma_X)
    res -= K * D * np.log(sigma_A)
    res -= D / 2 * det
    res -= 1 / (2 * sigma_X**2) * (trX - np.sum(l * uX))
    return res

def test_lp(func1, func2, func3, N, D, K, times, sigma_X = None, sigma_A = None):
    assert(D >= K)
    diff = np.zeros(times)
    diff[:] = np.nan
    time_arr = np.zeros((3, times))
    with tqdm(total=times) as pbar:
        for i in range(times):
            X = np.random.normal(0, 1, (N, D))
            Z = np.random.randint(0, 1, (N, K)) 
            Z = Z.astype('float')
            if sigma_X is None:
                sigma_X = np.random.random()
            if sigma_A is None:
                sigma_A = np.random.random()
            trX = np.trace(X.T @ X)
            time_start = time.time()
            ans1 = func1(trX = trX, X = X, Z = Z, sigma_X = sigma_X, sigma_A = sigma_A)
            time_arr[0, i] = time.time() - time_start
            time_start = time.time()
            ans2 = func2(trX = trX, X = X, Z = Z, sigma_X = sigma_X, sigma_A = sigma_A)
            time_arr[1, i] = time.time() - time_start
            time_start = time.time()
            ans3 = func3(trX = trX, X = X, Z = Z, sigma_X = sigma_X, sigma_A = sigma_A)
            time_arr[2, i] = time.time() - time_start
            diff[i] = max(np.abs(ans2 - ans1), np.abs(ans3 - ans1), np.abs(ans3 - ans2))
            pbar.update(1)
    
    return np.max(diff), np.mean(time_arr, axis = 1), np.max(time_arr, axis = 1)

def _print(diff):
    print("Maximum discrepancy:", format(diff[0], '.3e'))
    print("lp-test".rjust(8), "original".rjust(10), "improved".rjust(10), "numba".rjust(10))
    print("Mean.".rjust(8), format(diff[1][0], '.3e').rjust(10), format(diff[1][1], '.3e').rjust(10), format(diff[1][2], '.3e').rjust(10))
    print("Max.".rjust(8), format(diff[2][0], '.3e').rjust(10), format(diff[2][1], '.3e').rjust(10), format(diff[2][2], '.3e').rjust(10))


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "N:D:K:T:", ["sigma_X=","sigma_A="])
    sigma_X = None
    sigma_A = None
    for opt, arg in opts:
        if opt in ("-N"):
            N = int(arg)
        elif opt in ("-D"):
            D = int(arg)
        elif opt in ("-K"):
            K = int(arg)
        elif opt in ("-T"):
            T = int(arg)
        elif opt in ("--sigma_X"):
            sigma_X = float(arg)
        elif opt in ("--sigma_A"):
            sigma_A = float(arg)
    diff = test_lp(lp_original, lp, lp_numba, N, D, K, T, sigma_X, sigma_A)
    print(diff)