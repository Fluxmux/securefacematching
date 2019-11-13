import json
import requests
import logging
import datetime
import sys
import os
import base64

import numpy as np

from time import sleep
from random import random
from apscheduler.schedulers.background import BackgroundScheduler

from mpyc.runtime import mpc
from mpyc import thresha

logging.basicConfig(level=logging.DEBUG)
l = logging.getLogger("virtual_device")

# Suppress noisy loggers: apscheduler, urllib3
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def transmit_sample(host, port):
    """
    Transmits random JSON-encoded data sample
    to a given IP address and port number.

    host : IP address of host (string)
    port : Port number (int)
    """
    
    #read samples and split to secrets
    living_data = np.loadtxt('living_data.csv', delimiter=",")
    temperature = living_data[:, 0].tolist()
    airco_status = living_data[:, 1].tolist()
    
    secnum = mpc.SecFxp()
    #secnum = mpc.SecInt()
    
    temperature_sec_ = [secnum(a) for a in temperature]
    airco_status_sec_ = [secnum(a) for a in airco_status]
    stype = type(temperature_sec_[0])
    field = stype.field
    
    temperature_sec = [a.df for a in temperature_sec_] 
    airco_status_sec = [a.df for a in airco_status_sec_] 
    
    m = len(host)
    t = 1
    N = len(temperature)
    for i in np.arange(N):
        # Generate shares for each data sample in temperature and airco_status
        temperature_shares = [None] * m
        aircostatus_shares = [None] * m
        temperature_shares = thresha.random_split([temperature_sec[i]], t, m)
        aircostatus_shares = thresha.random_split([airco_status_sec[i]], t, m)
        
        temperature_shares_str = []
        for other_pid, data in enumerate(temperature_shares):
            data = field.to_bytes(data)
            temperature_shares_str.append(base64.b64encode(data).decode())
            
        aircostatus_shares_str = []
        for other_pid, data in enumerate(aircostatus_shares):
            data = field.to_bytes(data)
            aircostatus_shares_str.append(base64.b64encode(data).decode())
            
        print(temperature_shares_str)
        
        #send shares to MPC servers
        for j in np.arange(m):
            sample = {
                "timestamp" : datetime.datetime.now().isoformat(),
                "temperature" : temperature_shares_str[j],
                "airco" : aircostatus_shares_str[j]
            }
            sample_json = json.dumps(sample)
            l.debug(f"Sample: {sample_json}")
            r = requests.put(f"http://{host[j]}:{port[j]}/store", 
                             json=sample_json)
                             #verify=os.getenv("CERT_PATH"))
            l.debug(f"HTTP response code: {r.status_code}")


if __name__ == '__main__':
    bs = BackgroundScheduler({"apscheduler.timezone": "Europe/Brussels"})
    bs.start()
    # Transmit sample every second for 15 seconds
    # to target host and port
    host0 = os.getenv("SERVER_IP0")
    port0 = os.getenv("SERVER_PORT0")
    host1 = os.getenv("SERVER_IP1")
    port1 = os.getenv("SERVER_PORT1")
    host2 = os.getenv("SERVER_IP2")
    port2 = os.getenv("SERVER_PORT2")
    host = [host0, host1, host2]
    port = [port0, port1, port2]
    bs.add_job(transmit_sample, "interval", [host, port], seconds=1, id="ts_id")
    sleep(15)
    bs.remove_job("ts_id")
    bs.shutdown()