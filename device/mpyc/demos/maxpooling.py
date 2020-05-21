import numpy as np
import requests
from send_shares_mpcservers import send_shares_mpc

#MPC parties
hosts = ['LAP-BE-NIKOVIO1', 'LAP-BE-NIKOVIO1', 'LAP-BE-NIKOVIO1']
ports = [5000, 5001, 5002]

#clear data
info = []
#send input as flattened ndarray
for size in range(10,251,10):
    for j in np.arange(len(hosts)):
        #print(f'clearing datapart test on server {hosts[j]}:{ports[j]}')
        r = requests.get(f'http://{hosts[j]}:{ports[j]}/clear_all?datapart=test')

    image = np.random.rand(size,size)
    image = image.flatten()
    send_shares_mpc(image, ['Image'], 'test', hosts, ports, combined = True)

    url = f'http://{hosts[0]}:{ports[0]}/mpyc_launch?api=maxpooling_server'
    #compute
    #print(f'Sending request: {url}')
    response = requests.get(url)
    print(str(size) + ": " + str(response.text))
    info.append(response.text)
print(info)
