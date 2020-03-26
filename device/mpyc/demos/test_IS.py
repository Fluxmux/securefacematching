import numpy as np
import cv2
import requests
import os
from send_shares_mpcservers import send_shares_mpc

#MPC parties
hosts = ['LAP-BE-NIKOVIO1', 'LAP-BE-NIKOVIO1', 'LAP-BE-NIKOVIO1']
ports = [5000, 5001, 5002]

#clear data
for j in np.arange(len(hosts)):
    print(f'removing secret shares on server {hosts[j]}:{ports[j]}')
    r = requests.get(f'http://{hosts[j]}:{ports[j]}/clear_all?datapart=test')
    r = requests.get(f'http://{hosts[j]}:{ports[j]}/clear_all?datapart=model')

"""
image = cv2.imread(os.path.join('sharpen_image_data', 'mountain_view.jpg'), cv2.IMREAD_GRAYSCALE)
print(image.shape)
"""

filter_1 = np.array([[-1, -1, -1],
                     [ 0,  0,  0],
                     [ 1,  1,  1]])

filter_2 = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])
#construct testdata

image = np.array([[215, 175, 135, 215, 140],
                  [50,  145,  20,  40,  80],
                  [25,  155, 230, 185, 200],
                  [135, 210, 220, 240,  30],
                  [15,    0,  55, 150, 255]],
                  dtype=np.uint8)
"""
kernel = np.array([[-2, 1, 0],
                   [3, -4, 2],
                   [2, 0, -1]],
                    dtype=float)
"""
#send input as flattened ndarray
image = image.flatten()
filter_1 = filter_1.flatten()
filter_2 = filter_2.flatten()
filters = np.concatenate((filter_1, filter_2), axis=None)

#send_shares_mpc requires data and dataname to match
print(f'sending shares of input {image} to servers')
send_shares_mpc(image, ['Image'], 'test', hosts, ports, combined = True)

print(f'sending shares of input {filters} to servers')
send_shares_mpc(filters, ['Filters'], 'model', hosts, ports, combined = True)

url = f'http://{hosts[0]}:{ports[0]}/mpyc_launch?api=test_IS_server'

#compute
print(f'Sending request: {url}')
response = requests.get(url)
print(f'Response status code: {response.status_code}')
output = response.text
output = output.replace("[", "")
output = output.replace("]", "")
output = np.fromstring(output, dtype=float, sep=',').astype(int)
print(output)
output = np.clip(output, 0, 255)
output = np.reshape(output, (3,3))
print(output)
