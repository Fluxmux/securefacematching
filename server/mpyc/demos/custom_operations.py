import numpy as np
from mpyc.runtime import mpc
import pickle

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

def unpadded_correlation(image, kernel):
    hi, wi = image.shape
    hk, wk = kernel.shape

    out = []
    for row_img in np.arange(0, hi - 2):
        for col_img in np.arange(0, wi - 2):
            mul00 = kernel[0,0] * (image[row_img + 0][col_img + 0])
            mul01 = kernel[0,1] * (image[row_img + 0][col_img + 1])
            mul02 = kernel[0,2] * (image[row_img + 0][col_img + 2])
            mul10 = kernel[1,0] * (image[row_img + 1][col_img + 0])
            mul11 = kernel[1,1] * (image[row_img + 1][col_img + 1])
            mul12 = kernel[1,2] * (image[row_img + 1][col_img + 2])
            mul20 = kernel[2,0] * (image[row_img + 2][col_img + 0])
            mul21 = kernel[2,1] * (image[row_img + 2][col_img + 1])
            mul22 = kernel[2,2] * (image[row_img + 2][col_img + 2])
            sum = mpc.add(mul00, mul01)
            sum = mpc.add(sum, mul02)
            sum = mpc.add(sum, mul10)
            sum = mpc.add(sum, mul11)
            sum = mpc.add(sum, mul12)
            sum = mpc.add(sum, mul20)
            sum = mpc.add(sum, mul21)
            sum = mpc.add(sum, mul22)
            out.append(sum)
    out = np.asarray(out)
    out = np.reshape(out, (-1, image.shape[0] - 2))
    return out

def add_bias(image, bias):
    return np.add(image, bias)

def save(out, filename):
    pickle.dump(out, filename)
