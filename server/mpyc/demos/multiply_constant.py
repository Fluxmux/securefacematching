import numpy as np
from timeit import default_timer as timer
from mpyc.runtime import mpc
from load_database import load_data
#from LogisticRegressionMPC import LogisticRegressionMPC

def sqrt(n, lim=75):
    """
    Computes the sqrt of a number using Newton's method.
    Input:
        n : Float.
        lim : Newton's Method iteration limit (default=10).
    Output:
        sqrt: Square root of n.
    """
    m = n
    for _ in range(lim):
        m = 0.5*(m+(n/m))
    return m


async def main():

    await mpc.start()

    #load model
    print(f'loading model...')
    model = load_data('FALL_MODEL_LR', 'model', data_secrecy='cleartext')
    model = np.concatenate(model).tolist()
    model = [a.df for a in model]
    print(f'model: {model}')

#    #load test data
    print(f'loading data...')
    test_data = load_data('Fall', 'test', limitN=21)
    #print(f'test data: {test_data}')

    start = timer()
    secnum = mpc.SecFxp()
    const_value0 = secnum(0.0)
    const_value1 = secnum(1.0)
    const_value2 = secnum(2.0)

    test_value = await mpc.output(test_data[0][0])
    print(f'test data: {test_value}')

    print("$$$\n")
    print('{},'.format(await mpc.output(test_value*const_value0)/(2**16)))
    print('{},'.format(await mpc.output(test_value*const_value1)/(2**16)))
    print('{},'.format(await mpc.output(test_value*const_value2)/(2**16)))

    print("$$$")
    end = timer()
    running_time = end - start
    print(f'Run time: {running_time}')

    await mpc.shutdown()

    #return predictions


if __name__ == '__main__':
    mpc.run(main())
