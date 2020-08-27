import numpy as np
from tqdm import trange
import time
import sys
import getopt

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

def lp_optimization(X, Z, C, i, sigma_X = 1, sigma_A = 1):
    e = np.zeros(Z.shape[0])
    e[i] = 1
    trX = np.trace(X.T @ X)
    base = lp(trX, X, Z, sigma_X, sigma_A)
    diff = [0] * C
    for c in range(C):
        Z = np.c_[Z, e]
        diff[c] = lp(trX, X, Z, sigma_X, sigma_A) - base
    return diff

def lp_recursive(X, Z, C, i, sigma_X = 1, sigma_A = 1):
    D = X.shape[1]
    e = np.zeros(Z.shape[0])
    e[i] = 1
    U, S, _ = np.linalg.svd(Z, full_matrices = False)
    d = S ** 2 / (S ** 2 + sigma_X**2/sigma_A**2)
    coef = d * U[i,]
    gamma_i = U @ coef
    cur_diff = 0
    diff = [0] * C
    for c in range(C):
        mu = 1 + sigma_X**2/sigma_A**2 - gamma_i[i]
        t = 1/mu * np.sum((X.T @ gamma_i - X[i])**2)
        cur_diff += D * np.log(sigma_X / sigma_A)
        cur_diff -= D / 2 * np.log(mu) 
        cur_diff += t/(2 * sigma_X**2)
        gamma_i += 1/mu * (gamma_i[i] - 1) * (gamma_i - e)
        diff[c] = cur_diff
    return diff

def compare(N, D, K, C, i, sigma_X = 1, sigma_A = 1):
    X = np.random.normal(0, 1, (N, D))
    Z = np.random.randint(0, 2, (N, K))
    time_1 = time.time()
    opt = lp_optimization(X, Z, C, i, sigma_X, sigma_A)
    time_opt = time.time() - time_1
    time_2 = time.time()
    rec = lp_recursive(X, Z, C, i, sigma_X, sigma_A)
    time_rec = time.time() - time_2
    return np.array(opt) - np.array(rec), time_opt, time_rec

def multicompare(N, D, K, C, T):
    assert(D > K)
    res = np.zeros((3, T))
    for t in trange(T):
        i = np.random.randint(0, N)
        sigma_X = np.random.random()
        sigma_A = np.random.random()
        err, topt, trec = compare(N, D, K, C, i, sigma_X, sigma_A)
        res[0, t] = np.max(err)
        res[1, t] = topt
        res[2, t] = trec
    return np.mean(res[0]), np.mean(res[1]), np.max(res[1]), np.mean(res[2]), np.max(res[2])

def _print(diff):
    print("Maximum discrepancy:", format(diff[0], '.3e'))
    print("K-test".rjust(8), "original".rjust(10), "recursion".rjust(10))
    print("Mean.".rjust(8), format(diff[1], '.3e').rjust(10), format(diff[3], '.3e').rjust(10))
    print("Max.".rjust(8), format(diff[2], '.3e').rjust(10), format(diff[4], '.3e').rjust(10))

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "N:D:K:C:T:")
    for opt, arg in opts:
        if opt in ("-N"):
            N = int(arg)
        elif opt in ("-D"):
            D = int(arg)
        elif opt in ("-K"):
            K = int(arg)
        elif opt in ("-C"):
            C = int(arg)
        elif opt in ("-T"):
            T = int(arg)
    diff = multicompare(N, D, K, C, T)
    _print(diff)
