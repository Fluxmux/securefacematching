import numpy as np
import math
from mpyc.runtime import mpc
from load_database import load_data
from custom_operations import convolution, relu, maxpool
from timeit import default_timer as timer


async def main():
    print("Starting MPC")
    await mpc.start()
    print("MPC started")
    images = load_data('Image', 'test')
    kernels = load_data('Filters', 'model')
    image = images[0]
    kernel = kernels[0]
    print("Data loaded")

    image = np.reshape(image, (int(math.sqrt(len(image))), int(math.sqrt(len(image)))))
    kernel = np.reshape(kernel, (int(math.sqrt(len(kernel))), int(math.sqrt(len(kernel)))))
    print("Data reshaped")
    print(type(image), np.asarray(image).shape)
    print(type(kernel), np.asarray(kernel).shape)

    start = timer()

    start_conv = timer()
    conv = convolution(image, kernel)
    end_conv = timer()
    print(f'Run time: {end_conv - start_conv}')
    await mpc.barrier()
    start_maxp = timer()
    maxp = maxpool(conv)
    end_maxp = timer()
    print(f'Run time: {end_maxp - start_maxp}')
    await mpc.barrier()
    start_relu = timer()
    result = relu(maxp)
    end_relu = timer()
    print(f'Run time: {end_relu - start_relu}')

    end = timer()
    running_time = end - start
    print(f'Compute time: {running_time}')

    result = list(np.asarray(result).flatten())
    print(type(result), np.asarray(result).shape)

    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")

    end = timer()
    running_time = end - start
    print(f'Run time: {running_time}')

if __name__ == '__main__':
    mpc.run(main())
