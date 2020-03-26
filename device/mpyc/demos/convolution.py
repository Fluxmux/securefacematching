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


image = np.array([[215, 175, 135, 215, 140],
                  [50,  145,  20,  40,  80],
                  [25,  155, 230, 185, 200],
                  [135, 210, 220, 240,  30],
                  [15,    0,  55, 150, 255]],
                  dtype=np.uint8)

m, n = image.shape

kernel = np.array([[1, 0, 0],
                   [0, 5, 0],
                   [0 , 0, -1]],
                    dtype=float)

#send input as flattened ndarray
image = image.flatten()
kernel = kernel.flatten()

#send_shares_mpc requires data and dataname to match
print(f'sending shares of input {image} to servers')
send_shares_mpc(image, ['Image'], 'test', hosts, ports, combined = True)

print(f'sending shares of input {kernel} to servers')
send_shares_mpc(kernel, ['Filters'], 'model', hosts, ports, combined = True)

url = f'http://{hosts[0]}:{ports[0]}/mpyc_launch?api=convolution_server'

#compute
print(f'Sending request: {url}')
response = requests.get(url)
print(f'Response status code: {response.status_code}')

output = response.text
output = output.replace("[", "")
output = output.replace("]", "")
output = np.fromstring(output, dtype=float, sep=',').astype(int)
output = np.clip(output, 0, 255)
output = np.reshape(output, (m,n))
print(output)