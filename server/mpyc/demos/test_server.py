import numpy as np
from mpyc.runtime import mpc
from load_database import load_data
from custom_operations import convolution

async def main():
    await mpc.start()
    image = load_data('Image', 'test')
    kernel = load_data('Kernel', 'model')
    image = np.reshape(image, (-1, 5))
    kernel = np.reshape(kernel, (-1, 3))

    result = convolution(image, kernel)

    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")

if __name__ == '__main__':
    mpc.run(main())
