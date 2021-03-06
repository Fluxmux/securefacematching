import numpy as np
from timeit import default_timer as timer
from mpyc.runtime import mpc
from load_database import load_data
from copy import deepcopy
import math

def unpadded_correlation(image, kernel):
    hi, wi = image.shape
    hk, wk = kernel.shape

    out = []
    print("***unpadded_correlation***")
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
    out = np.reshape(out, (-1, 98))
    return out


def sharpen_img(image, kernels):
    """ If we use unpadded_correlation we need to reduce the size of the image """
    new_image = image[1:99, 1:99]
    for kernel in kernels:
        new_image = mpc.matrix_add(new_image, unpadded_correlation(image, kernel))
    new_image = np.array(new_image).flatten()
    return new_image


async def main():
    await mpc.start()
    #load test data
    print(f'loading data...')
    data = load_data('Face', 'test')
    hor_filter = load_data('Hor_filter', 'test')
    ver_filter = load_data('Ver_filter', 'test')
    #data = np.array(data)
    data =
    hor_filter = np.reshape(hor_filter, (-1, 3))

    ver_filter = np.reshape(ver_filter, (-1, 3))
    start = timer()

    result = sharpen_img(data, (hor_filter, ver_filter)).tolist()
    print(type(result), len(result), type(result[0]))

    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")

    end = timer()
    running_time = end - start
    print(f'MPC total time: {running_time}')

if __name__ == '__main__':
    mpc.run(main())
