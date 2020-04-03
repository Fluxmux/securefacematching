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


def maxpool(X):
    print('---------- maxpooling ----------')
    # maxpooling 2 * 2 squares in images of size m * n with stride 2
    m, n = dim(X)
    Y = [None] * (m // 2)
    for i in range(0, m - 1, 2):
        Y[int(i/2)] = [None] * (n // 2)
        for j in range(0, n - 1, 2):
            Y[int(i/2)][int(j/2)] = mpc.max(X[i][j], X[i][j+1], X[i+1][j], X[i+1][j+1])
    return np.array(Y)


def relu(x):
    print("---------- relu ----------")
    return np.vectorize(lambda a: (a >= 0) * a)(x)
