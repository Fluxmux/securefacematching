import numpy as np
import math
from mpyc.runtime import mpc
from load_database import load_data
from custom_operations import convolution, relu, maxpool, unpadded_correlation, add_bias
from timeit import default_timer as timer
from decimal import *


async def main():
    print("Starting MPC")
    await mpc.start()
    print("MPC started")
    images = load_data('Face', 'test')
    image = images[0]
    print("Data loaded")

    image = np.reshape(image, (int(math.sqrt(len(image))), int(math.sqrt(len(image)))))
    print("Data reshaped")
    kernel = np.array([[-0.21190501749515533,-0.23755072057247162,0.19877947866916656],
                        [0.24508745968341827,-0.26056352257728577,0.15691310167312622],
                        [0.3300827145576477,-0.18680880963802338,-0.04544968530535698]])
    #bias = -0.09444483369588852

    start = timer()

    start_conv = timer()
    conv = unpadded_correlation(image, kernel)
    end_conv = timer()
    print(f'Conv time: {end_conv - start_conv}')

    """
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
    """
    end = timer()
    running_time = end - start
    print(f'Compute time: {running_time}')
    result = conv
    np.save("ul.npy", result)
    #result = list(np.asarray(result).flatten())

    print("$$$\n")
    #print(await mpc.output(result))
    print("OK")
    print("$$$")

if __name__ == '__main__':
    mpc.run(main())
