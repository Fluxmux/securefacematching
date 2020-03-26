from mpyc.runtime import mpc
from load_database import load_data

async def main():
    print("$$$$$$$$$$$$$")
    await mpc.start()
    print("#############")
    data = load_data('Numbers', 'test')
    flattened_data = [item for sublist in data for item in sublist]
    a = flattened_data[0]
    b = flattened_data[1]
    print("a:", type(a), a)
    print("b:", type(b), b)
    result = mpc.add(a, b)
    print("outputting result of type:", type(result), result)
    print("OUTPUT:", await mpc.output(result))
    print("$$$\n")
    print(await mpc.output(result))
    print("$$$")
    print("result ouputted")

if __name__ == '__main__':
    mpc.run(main())
