import numpy as np
import os
from matplotlib.image import imread
import matplotlib.pyplot as plt
import cv2
from math import ceil


hor_filter = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]])
print(type(hor_filter))

hor_filter_as_list = hor_filter.tolist()
print(type(hor_filter_as_list))
print(hor_filter)
print(hor_filter_as_list)

#def explicit_correlation(image, kernel):
#    hi, wi= image.shape
#    hk, wk = kernel.shape
#    image_padded = np.zeros(shape=(hi + hk - 1, wi + wk - 1))
#    image_padded[hk//2:ceil(-hk/2), wk//2:ceil(-wk/2)] = image
#    out = np.zeros(shape=image.shape)
#    for row in np.arange(0, hi-1):
#        for col in np.arange(0, wi-1):
#            out[row, col] = np.dot(kernel.reshape(1, -1), image_padded[row:row + 3, col:col + 3].reshape(1, -1).T)
#    return out
#
#
#image = imread(os.path.join('data_faces', '1.pgm'))
#image = cv2.resize(image, dsize=(100, 100))
#
#hor_filter = np.array([[-1, -1, -1],
#                       [ 0,  0,  0],
#                       [ 1,  1,  1]])
#
#ver_filter = np.array([[-1, 0, 1],
#                       [-1, 0, 1],
#                       [-1, 0, 1]])
#
#print("ORIGINAL")
#plt.gray()
#plt.figure()
#plt.imshow(image) 
#plt.show()
#
#print("AFTER horizontal filter")
#hor_image = explicit_correlation(image, hor_filter)
#plt.figure()
#plt.imshow(hor_image) 
#plt.show()
#
#print("AFTER vertical filter")
#ver_image = explicit_correlation(image, ver_filter)
#plt.figure()
#plt.imshow(ver_image) 
#plt.show()
#
#print("AFTER sum vertical and horizontal (high frequency response)")
#hfr_image = 0.2*np.add(hor_image, ver_image)
#hfr_image = np.clip(hfr_image, 0, 255)
#plt.figure()
#plt.imshow(hfr_image) 
#plt.show()
#
#print("FINAL RESULT")
#out = np.add(image, hfr_image)
#out = np.clip(out, 0, 255)
#plt.figure()
#plt.imshow(out) 
#plt.show()


