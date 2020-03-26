import numpy as np
import requests
from send_shares_mpcservers import send_shares_mpc

#MPC parties
hosts = ['LAP-BE-NIKOVIO1', 'LAP-BE-NIKOVIO1', 'LAP-BE-NIKOVIO1']
ports = [5000, 5001, 5002]

#clear data
for j in np.arange(len(hosts)):
    print(f'clearing datapart test on server {hosts[j]}:{ports[j]}')
    r = requests.get(f'http://{hosts[j]}:{ports[j]}/clear_all?datapart=test')

#send input as flattened ndarray
numbers = np.array([-5, 3])

print(f'sending shares of input {numbers} to servers')
send_shares_mpc(numbers, ['Numbers'], 'test', hosts, ports, combined = True)

url = f'http://{hosts[0]}:{ports[0]}/mpyc_launch?api=sum_server'
#compute
print(f'Sending request: {url}')
response = requests.get(url)
print(f'Response status code: {response.status_code}')
print(response.text)
output = int(float(response.text))
print(output)
