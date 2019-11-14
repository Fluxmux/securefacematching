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


def padded_correlation(image, kernel):
    hi, wi= image.shape
    hk, wk = kernel.shape

    image_padded = [[0 for x in range(0,hi+hk)] for y in range(0,wi+wk)]

    img_x_s = hk//2
    img_x_e = math.ceil(-hk/2)
    img_y_s = wk//2
    img_y_e = math.ceil(-wk/2)

    for x in np.arange(0, hi):
        for y in np.arange(0, wi):
            image_padded[x + img_x_s][y +img_y_s] = image[x, y]

    """
    PROBLEM with type conversion
    for x in np.arange(0, hi):
        for y in np.arange(0, wi):
            image_padded[x + img_x_s, y + img_y_s] = image[x, y]
    """

    out = np.zeros(shape=image.shape)
    for row_img in np.arange(0, hi-1):
        for col_img in np.arange(0, wi-1):
            for row_kernel in np.arange(0, hk):
                for col_kernel in np.arange(0, wk):
                    out[row_img, col_img] += kernel[row_kernel, col_kernel] * image_padded[row_img + row_kernel][col_img + col_kernel]

    for i in range(0,5):
        for j in range(0,5):
            print(out[i,j], type(out[i,j]))
    return out


def sharpen_img(image, kernels):
    """ If we use unpadded_correlation we need to reduce the size of the image """
    new_image = image[1:99, 1:99]
    for kernel in kernels:
        new_image = mpc.matrix_add(new_image, unpadded_correlation(image, kernel))
    new_image = np.array(new_image).flatten()
    return new_image


async def main():
    start = timer()
    await mpc.start()
    end = timer()
    running_time = end - start
    print(f'MPC start time: {running_time}')

    #load test data
    start = timer()
    print(f'loading data...')
    data = load_data('Face', 'test')
    data = np.reshape(data, (-1, 100))

    hor_filter = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]])

    ver_filter = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])

    result = sharpen_img(data, (hor_filter, ver_filter)).tolist()
    print(type(result), len(result), type(result[0]))
    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")
    print("result ouputted")


if __name__ == '__main__':
    mpc.run(main())
