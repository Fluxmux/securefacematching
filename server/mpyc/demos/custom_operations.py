import numpy as np
from mpyc.runtime import mpc

def dim(x):
    if isinstance(x, np.ndarray):
        s = list(x.shape)
    else:
        s = []
        while isinstance(x, list):
            s.append(len(x))
            x = x[0]
    return s


def convolution(X, W):
    print("---------- convolution ----------")
    m, n = dim(X)
    s = len(W) # s * s filter W
    s2 = (s - 1) // 2
    Y = [None] * m
    for i in range(m):
        Y[i] = [None] * n
        for j in range(n):
            t = 0
            ix = i - s2
            for di in range(s):
                if 0 <= ix < m:
                    jx = j - s2
                    for dj in range(s):
                        if 0 <= jx < n:
                            t += X[ix][jx] * W[di][dj]
                        jx += 1
                ix += 1
            Y[i][j] = t
    return Y

def relu(x):
    print("---------- relu ----------")
    return np.vectorize(lambda a: (a >= 0) * a)(x)
