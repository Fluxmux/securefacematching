import numpy as np
import math
from mpyc.runtime import mpc
from load_database import load_data
from custom_operations import layer
from timeit import default_timer as timer
import pickle


async def main():
    print("Starting MPC...")
    await mpc.start()
    print("MPC started")

    print("Loading data...")
    images = load_data("Face", "test")
    image = images[0]
    with open("parameters/weights.pkl", "rb") as f:
        weights = pickle.load(f)

    with open("parameters/biases.pkl", "rb") as f:
        biases = pickle.load(f)
    print("Data loaded")

    image = np.reshape(image, (int(math.sqrt(len(image))), int(math.sqrt(len(image)))))
    print("Data reshaped")

    start = timer()

    print("---------- LAYER 0 ----------")
    inputs_1 = layer(image, weights[0], biases[0])
    await mpc.barrier()

    print("---------- LAYER 1 ----------")
    inputs_2 = layer(inputs_1, weights[1], biases[1])
    await mpc.barrier()

    print("---------- LAYER 2 ----------")
    result = layer(inputs_2, weights[2], biases[2])
    await mpc.barrier()

    print("---------- FLATTENING ----------")
    result = list(np.asarray(result).flatten())

    end = timer()
    compute_time = end - start
    print(f"Compute time: {compute_time}")

    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")


if __name__ == "__main__":
    mpc.run(main())
