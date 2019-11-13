import numpy as np
from timeit import default_timer as timer
from mpyc.runtime import mpc
from load_database import load_data
import math

def correlation(image, kernel):
    hi, wi= image.shape
    hk, wk = kernel.shape
    image_padded = np.zeros(shape=(hi + hk - 1, wi + wk - 1))
    img_x_s = hk//2
    img_x_e = math.ceil(-hk/2)
    img_y_s = wk//2
    img_y_e = math.ceil(-wk/2)
    print(type(img_x_s),type(img_x_e),type(img_y_s),type(img_y_e))
    print(type(image), type(image[0,0]))
#    image_padded[img_x_s:img_x_e, img_y_s:img_y_e] = image
    for x in np.arange(img_x_s, img_x_e):
        for y in np.arange(img_y_s, img_y_e):
            image_padded[x, y] = image[x - 1, y - 1]
    out = np.zeros(shape=image.shape)
    for row in np.arange(0, hi-1):
        for col in np.arange(0, wi-1):
            #QUESTION: kernel is a list of non-secure integers how to do a product of constant and secure number
            out[row, col] = mpc.in_prod(np.concatenate(kernel).tolist(), np.concatenate(image_padded[row:row + 3, col:col +3]).tolist())
    return out


def sharpen_img(image, kernels):
    for kernel in kernels:
        image = np.add(image, correlation(image, kernel))
    return image


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
    hor_filter = load_data('Hor_filter', 'test')
    hor_filter = np.reshape(hor_filter, (-1, 3))
    ver_filter = load_data('Ver_filter', 'test')
    ver_filter = np.reshape(ver_filter, (-1, 3))

    """
    hor_filter = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]])

    ver_filter = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])
    """
    result = sharpen_img(data, (hor_filter, ver_filter))
    print("outputting result", await mpc.output(result), "of type:", type(result))
    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")
    print("result ouputted")


if __name__ == '__main__':
    mpc.run(main())
