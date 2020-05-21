from mpyc.runtime import mpc
from timeit import default_timer as timer
from load_database import load_data
from custom_operations import relu
import numpy as np
import math

async def main():
    await mpc.start()
    image = load_data('Image', 'test')[0]
    #image = np.reshape(image, (int(math.sqrt(len(image))), int(math.sqrt(len(image)))))

    #start = timer()
    #result = relu(image)
    #result = list(np.asarray(result).flatten())
    #end = timer()
    #running_time = end - start
    #print(f'Run time: {running_time}')
    start = timer()
    print(await mpc.output(image))
    end = timer()
    running_time = end - start
    print("$$$\n")
    print(running_time)
    print("$$$")


if __name__ == '__main__':
    mpc.run(main())
