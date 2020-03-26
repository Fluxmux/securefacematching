import numpy as np
from mpyc.runtime import mpc

"""
 function:  2D convolution without padding
 input:     image of size wi*hi (list)
            kernel of size s*s  (list)
 output:    1D list of convolved image
"""

def convolution(image, kernel):
    wi, hi = np.asarray(image).shape
    s, s = np.asarray(kernel).shape
    result = []
    k = s//2
    for x in range(k, wi - k):
        for y in range(k, hi - k):
            tmp_image = image[x-k:x+k+1][y-k:y+k+1]
            I = np.asarray(tmp_image).flatten().tolist()
            K = np.asarray(kernel).flatten().tolist()
            result.append(mpc.in_prod(I, K))
    return result
