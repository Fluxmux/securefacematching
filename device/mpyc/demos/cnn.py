import os
import matplotlib.pyplot as plt
import numpy as np
import requests
import cv2
from timeit import default_timer as timer
from send_shares_mpcservers import send_shares_mpc
from PIL import Image
import torchvision.transforms as transforms
import torch

#MPC parties
hosts = ['LAP-BE-NIKOVIO1', 'LAP-BE-NIKOVIO1', 'LAP-BE-NIKOVIO1']
ports = [5000, 5001, 5002]

#clear data
for j in np.arange(len(hosts)):
    print(f'clearing datapart test on server {hosts[j]}:{ports[j]}')
    r = requests.get(f'http://{hosts[j]}:{ports[j]}/clear_all?datapart=test')

#send image as flattened ndarray
size = 55
transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
image = Image.open(os.path.join('data_faces', '1.pgm'))
image = image.convert("L")
image = transform(image)
image = image.to(torch.device("cpu"))
image = image[None, ...]
image = np.array(image)[0][0]

flattened_image = image.flatten().tolist()

print(f'sending shares of input {flattened_image} to servers')
send_shares_mpc(flattened_image, ['Face'], 'test', hosts, ports, combined=True)

#compute
url = f'http://{hosts[0]}:{ports[0]}/mpyc_launch?api=cnn_server'
print(f'Sending request: {url}')
response = requests.get(url)

print(f'Response status code: {response.status_code}')
output = response.text

print(output)
