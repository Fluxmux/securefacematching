import os
from matplotlib.image import imread
import numpy as np
import requests
import cv2
from send_shares_mpcservers import send_shares_mpc

#MPC parties
hosts = ['LAP-BE-NIKOVIO1', 'LAP-BE-NIKOVIO1', 'LAP-BE-NIKOVIO1'] #LAP-BE-shephca1.vasco.com
ports = [5000, 5001, 5002]

#clear data
for j in np.arange(len(hosts)):
    print(f'clearing datapart test on server {hosts[j]}:{ports[j]}')
    r = requests.get(f'http://{hosts[j]}:{ports[j]}/clear_all?datapart=test')

#send image as flattened ndarray
image = imread(os.path.join('data_faces', '1.pgm'))
image = cv2.resize(image, dsize=(100, 100))
flattened_image = image.flatten()

hor_filter = np.array([[-1, -1, -1],
                       [ 0,  0,  0],
                       [ 1,  1,  1]]).flatten()


ver_filter = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]]).flatten()

print(f'sending shares of input {hor_filter} to servers')
send_shares_mpc(hor_filter, ['Hor_filter'], 'test', hosts, ports, combined=True)
print(f'sending shares of input {ver_filter} to servers')
send_shares_mpc(ver_filter, ['Ver_filter'], 'test', hosts, ports, combined=True)
print(f'sending shares of input {flattened_image} to servers')
send_shares_mpc(flattened_image, ['Face'], 'test', hosts, ports, combined=True)

#compute
url = f'http://{hosts[0]}:{ports[0]}/mpyc_launch?api=image_sharpening_server'
print(f'Sending request: {url}')
response = requests.get(url)
print(f'Response status code: {response.status_code}')
output = response.text
print(type(output))
print(output)
