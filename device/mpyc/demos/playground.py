import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import math
from math import ceil

print(np.asarray([[1, 2, 3]]).shape)
"""
#construct testdata
image = np.array([[215, 175, 135, 215, 140],
                  [50,  145,  20,  40,  80],
                  [25,  155, 230, 185, 200],
                  [135, 210, 220, 240,  30],
                  [15,    0,  55, 150, 255]],
                  dtype=float)

kernel = np.array([[-2, 1, 0],
                   [3, -4, 2],
                   [2, 0, -1]],
                    dtype=float)
"""
"""
plaintext = cv2.imread('plaintext_out.png', cv2.IMREAD_GRAYSCALE)
secure = cv2.imread('secure_out.png', cv2.IMREAD_GRAYSCALE)

h, w = secure.shape

wrong = 0
correct = 0
for i in range(0,h):
    for j in range(0,w):
        if secure[i][j] != plaintext[i][j]:
            print(secure[i][j], plaintext[i][j])
            print("x:", i, "y:", j)
            wrong += 1
        else:
            correct += 1

print("\n*********************")
print("correct:", correct)
print("wrong:", wrong)
print("*********************")

"""



def explicit_correlation(image, kernel):
    hi, wi= image.shape
    hk, wk = kernel.shape
    image_padded = np.zeros(shape=(hi + hk - 1, wi + wk - 1))
    image_padded[hk//2:ceil(-hk/2), wk//2:ceil(-wk/2)] = image
    out = np.zeros(shape=image.shape)
    for row in np.arange(0, hi-1):
        for col in np.arange(0, wi-1):
            out[row, col] = np.dot(kernel.reshape(1, -1), image_padded[row:row + 3, col:col + 3].reshape(1, -1).T)
    return out

def correlation(image, kernel):
    hi, wi= image.shape
    hk, wk = kernel.shape
    image_padded = np.zeros(shape=(hi + hk - 1, wi + wk - 1))
    img_x_s = hk//2
    img_x_e = math.ceil(-hk/2)
    img_y_s = wk//2
    img_y_e = math.ceil(-wk/2)

    print(img_x_s,img_x_e,img_y_s,img_y_e)

    for x in np.arange(0, hi):
        for y in np.arange(0, wi):
            image_padded[x + img_x_s, y +img_y_s] = image[x, y]
    out = np.zeros(shape=image.shape)

    for row_img in np.arange(0, hi-1):
        for col_img in np.arange(0, wi-1):
            for row_kernel in np.arange(0, hk):
                for col_kernel in np.arange(0, wk):
                    out[row_img, col_img] += kernel[row_kernel, col_kernel] * image_padded[row_img + row_kernel, col_img + col_kernel]
    return out


result = correlation(image, kernel)
print(result)
"""
"""

image = cv2.imread(os.path.join('sharpen_image_data', 'mountain_view.jpg'), cv2.IMREAD_GRAYSCALE)
print("shape of image:", image.shape)

hor_filter = np.array([[-1, -1, -1],
                       [ 0,  0,  0],
                       [ 1,  1,  1]])

ver_filter = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])

print("ORIGINAL")
plt.gray()
plt.figure()
plt.imshow(image)
plt.show()

print("AFTER horizontal filter")
hor_image = correlation(image, hor_filter)
plt.figure()
plt.imshow(hor_image)
plt.show()

print("AFTER vertical filter")
ver_image = correlation(image, ver_filter)
plt.figure()
plt.imshow(ver_image)
plt.show()

print("AFTER sum vertical and horizontal (high frequency response)")
hfr_image = np.add(hor_image, ver_image)
plt.figure()
plt.imshow(hfr_image)
plt.show()

print("FINAL RESULT")
out = np.add(image, 0.2*hfr_image)
out = np.clip(out, 0, 255)
plt.figure()
plt.imshow(out)
plt.show()
out = out.astype(np.uint8)

cv2.imwrite('plaintext_out.jpg', out[1:359, 1:639])
