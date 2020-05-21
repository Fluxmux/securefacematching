from mpyc.runtime import mpc
from timeit import default_timer as timer
from load_database import load_data
from custom_operations import convolution
import numpy as np
import math

async def main():
    await mpc.start()
    image = load_data('Image', 'test')[0]
    kernel = load_data('Kernel', 'model')[0]
    image = np.reshape(image, (int(math.sqrt(len(image))), int(math.sqrt(len(image)))))
    kernel = np.reshape(kernel, (int(math.sqrt(len(kernel))), int(math.sqrt(len(kernel)))))

    start = timer()
    result = convolution(image, kernel)
    end = timer()
    running_time = end - start
    print(f'Run time: {running_time}')

    print("$$$\n")
    print(running_time)
    print("$$$")

if __name__ == '__main__':
    mpc.run(main())
