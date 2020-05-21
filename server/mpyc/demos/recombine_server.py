import numpy as np
from mpyc.runtime import mpc
import pickle


async def main():
    print("Starting MPC")
    await mpc.start()

    result = np.load("ul.pkl")
    result = list(np.asarray(result).flatten())

    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")

if __name__ == '__main__':
    mpc.run(main())
