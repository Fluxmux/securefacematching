from mpyc.runtime import mpc
from timeit import default_timer as timer
from load_database import load_data
from custom_operations import relu

async def main():
    await mpc.start()
    data = load_data('Numbers', 'test')
    flattened_data = [item for sublist in data for item in sublist]
    a = flattened_data[0]
    print("a:", type(a), a)
    start = timer()
    result = relu(a)

    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")

    end = timer()
    running_time = end - start
    print(f'Run time: {running_time}')

if __name__ == '__main__':
    mpc.run(main())
