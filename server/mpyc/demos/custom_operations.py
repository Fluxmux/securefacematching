import numpy as np
from mpyc.runtime import mpc


def dim(x):
    """Return dimension of iteratable object independent of type

    Arguments:
        x {np.ndarray or list} -- iteratable object

    Returns:
        list -- shape of iteratable object
    """
    if isinstance(x, np.ndarray):
        s = list(x.shape)
    else:
        s = []
        while isinstance(x, list):
            s.append(len(x))
            x = x[0]
    return s


def convolution(X, W):
    """Single padded correlation on input X with filter weights W

    Arguments:
        X {np.ndarray or list} -- input or image
        W {np.ndarray or list} -- weights of filter/kernel

    Returns:
        list -- output of padded correlation function (has same shape as input X)
    """
    print("---------- convolution ----------")
    m, n = dim(X)
    s = len(W)  # s * s filter W
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
    """Max pooling function for subsampling an input based on maximal values

    Arguments:
        X {np.ndarray} -- input or image that needs to be downsampled

    Returns:
        np.ndarray -- downsampled output
    """
    print("---------- maxpooling ----------")
    # maxpooling 2 * 2 squares in images of size m * n with stride 2
    m, n = dim(X)
    Y = [None] * (m // 2)
    for i in range(0, m - 1, 2):
        Y[int(i / 2)] = [None] * (n // 2)
        for j in range(0, n - 1, 2):
            Y[int(i / 2)][int(j / 2)] = mpc.max(
                X[i][j], X[i][j + 1], X[i + 1][j], X[i + 1][j + 1]
            )
    return np.array(Y)


def relu(x):
    """Activation function: Rectified Linear Unit (ReLU)

    Arguments:
        x {np.ndarray or list} -- input that needs to be activated

    Returns:
        np.ndarray or list -- output of activation function over input
    """
    print("---------- relu ----------")
    return np.vectorize(lambda a: (a >= 0) * a)(x)


def unpadded_correlation(image, kernel):
    """Single unpadded correlation function over image as input and with kernel as weights

    Arguments:
        image {np.ndarray} -- input of correlation function
        kernel {np.ndarray} -- weights of filter/kernel

    Returns:
        np.ndarray -- convoluted output
    """
    hi, wi = image.shape
    hk, wk = kernel.shape

    out = []
    for row_img in np.arange(0, hi - 2):
        for col_img in np.arange(0, wi - 2):
            mul00 = kernel[0, 0] * (image[row_img + 0][col_img + 0])
            mul01 = kernel[0, 1] * (image[row_img + 0][col_img + 1])
            mul02 = kernel[0, 2] * (image[row_img + 0][col_img + 2])
            mul10 = kernel[1, 0] * (image[row_img + 1][col_img + 0])
            mul11 = kernel[1, 1] * (image[row_img + 1][col_img + 1])
            mul12 = kernel[1, 2] * (image[row_img + 1][col_img + 2])
            mul20 = kernel[2, 0] * (image[row_img + 2][col_img + 0])
            mul21 = kernel[2, 1] * (image[row_img + 2][col_img + 1])
            mul22 = kernel[2, 2] * (image[row_img + 2][col_img + 2])
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
    """Adds the bias to all elements of 2D vector

    Arguments:
        image {np.ndarray} -- 2D input matrix
        bias {int} -- value that is added

    Returns:
        np.ndarray -- output after adding the bias to the input
    """
    return np.add(image, bias)


def layer(inputs, weights, biases, in_channels, out_channels):
    """One full CNN layer containing multiple convolutions on multiple images + adding bias + max pooling + ReLU activation function (in this order)

    Arguments:
        inputs {list} -- list of input images for certain layer
        weights {list} -- list weights for certain layer
        biases {list} -- list of biases for certain layer
        in_channels {int} -- number of input channels of certain layer
        out_channels {int} -- number of output channels of certain layer

    Returns:
        outputs {list} -- list of output images for certain layer
    """
    outputs = []
    for o in range(out_channels):
        bias = biases[o]
        output = unpadded_correlation(inputs[0], weights[0][o])
        for i in range(1,in_channels):
            output += unpadded_correlation(inputs[i], weights[i][o])
        output = add_bias(output, bias)
        outputs.append(output)
    return outputs