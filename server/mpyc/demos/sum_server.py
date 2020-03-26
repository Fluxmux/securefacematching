from mpyc.runtime import mpc
from timeit import default_timer as timer
from load_database import load_data

async def main():
    await mpc.start()
    data = load_data('Numbers', 'test')
    flattened_data = [item for sublist in data for item in sublist]
    a = flattened_data[0]
    b = flattened_data[1]
    print("a:", type(a), a)
    print("b:", type(b), b)
    start = timer()
    result = mpc.add(a, b)

    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")

    end = timer()
    running_time = end - start
    print(f'Run time: {running_time}')

if __name__ == '__main__':
    mpc.run(main())
