import numpy as np
from timeit import default_timer as timer
from mpyc.runtime import mpc
from load_database import load_data
from copy import deepcopy
import math

x = 100
y = 100

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
    #out = np.reshape(out, (638, 358))
    out = np.reshape(out, (x-2, y-2))
    return out

def sharpen_img(image, kernels):
    """ If we use unpadded_correlation we need to reduce the size of the image """
    #new_image = image[1:639, 1:359]
    new_image = image[1:(x-1), 1:(y-1)]
    for kernel in kernels:
        new_image = mpc.matrix_add(new_image, unpadded_correlation(image, kernel))
    new_image = np.array(new_image).flatten()
    return new_image


async def main():
    start = timer()
    await mpc.start()
    end = timer()
    running_time = end - start
    #print(f'MPC start time: {running_time}')
    #load test data
    start = timer()
    #print(f'loading data...')
    data = load_data('Face', 'test')
    #print("data type:", type(data))
    data = np.array(data)
    #data = np.reshape(data, (640, 360))
    data = np.reshape(data, (x, y))

    hor_filter = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]])

    ver_filter = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])

    with open('/tmp/1.txt', 'w') as f:
        f.write("line 74\n")
    f.close()

    result = sharpen_img(data, (hor_filter, ver_filter)).tolist()
    #print(type(result), len(result), type(result[0]))
    with open('/tmp/2.txt', 'w') as f:
        f.write("line 79\n")
    f.close()
    print("$$$\n")
    output = await mpc.output(result)
    """
    with open('/tmp/result.txt', 'w') as f:
        f.write(str(output))
    f.close()
    """
    print(output)
    print("$$$")
    #print("result ouputted")

if __name__ == '__main__':
    mpc.run(main())
