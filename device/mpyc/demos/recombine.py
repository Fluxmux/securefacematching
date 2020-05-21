import requests
from send_shares_mpcservers import send_shares_mpc
url = "http://LAP-BE-NIKOVIO1:5000/mpyc_launch?api=recombine"
response = requests.get(url)
print(response.text)
