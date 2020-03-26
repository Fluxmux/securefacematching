import numpy as np
from mpyc.runtime import mpc
from load_database import load_data
from custom_operations import convolution, relu

async def main():
    print("Starting MPC")
    await mpc.start()
    print("MPC started")
    images = load_data('Image', 'test')
    kernels = load_data('Filters', 'model')
    image = images[0]
    kernel = kernels[0]
    print("Data loaded")

    image = np.reshape(image, (5, 5))#.tolist()
    kernel = np.reshape(kernel, (3, 3))#.tolist()
    print("Data reshaped")
    print(type(image), np.asarray(image).shape)
    print(type(kernel), np.asarray(kernel).shape)

    print("start convolution")
    conv = convolution(image, kernel)
    await mpc.barrier()
    result = relu(conv)

    result = list(np.asarray(result).flatten())
    print(type(result), np.asarray(result).shape)

    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")

if __name__ == '__main__':
    mpc.run(main())
