import numpy as np
from mpyc.runtime import mpc
from load_database import load_data
from custom_operations import convolution

async def main():
    print("Starting MPC")
    await mpc.start()
    print("MPC started")
    image = load_data('Image', 'test')
    kernels = load_data('Filters', 'model')
    image = image[0]
    _image = image[1:5][1:5]
    _image = np.asarray(_image).flatten().tolist()
    kernel_1 = kernels[0][0:9]
    kernel_2 = kernels[0][9:18]
    print(kernel_1)
    print("Data loaded")
    image = np.reshape(image, (5, 5)).tolist()
    kernel_1 = np.reshape(kernel_1, (3, 3)).tolist()
    kernel_2 = np.reshape(kernel_2, (3, 3)).tolist()
    print("Data reshaped")

    print("convolution 1")
    filter_1 = convolution(image, kernel_1)
    print("convolution 2")
    filter_2 = convolution(image, kernel_2)

    print(np.asarray(filter_1).shape,np.asarray([filter_2]).shape, np.asarray([_image]).shape)
    filter = mpc.matrix_add([filter_1], [filter_2])
    print("addition 2")
    print(type(image), type(filter), np.asarray([filter]).shape)
    result = mpc.matrix_add([_image], [filter])
    result = list(np.asarray(result).flatten())
    print(type(result), np.asarray(result).shape)
    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")

if __name__ == '__main__':
    mpc.run(main())
