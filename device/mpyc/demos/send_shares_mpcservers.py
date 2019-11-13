import json
import requests
import datetime
import sys
import os
import base64
import socket
import asyncio
import time
from timeit import default_timer as timer
from joblib import Parallel, delayed

import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
print(os.path.dirname(os.getcwd()))

from mpyc.runtime import mpc
from mpyc import thresha

running_time_compute_share = 0
running_time_upload_share = 0

def get_timings():
    return running_time_compute_share, running_time_upload_share

def send_data(data, datapart, host, port):
     #send shares to MPC servers
    #print(f"Sample: {data}")
    r = requests.put(f"http://{host}:{port}/store?datapart={datapart}", 
                    json=data)
                    #verify=os.getenv("CERT_PATH"))
    #print(f"HTTP response code: {r.status_code}")
    

def send_shares_mpc_single(data, dataname, datapart, hosts, ports):
    #print(f'Sending each data sample...')
    if data.ndim == 1:
        cols = data.shape[0]
        rows = 1
        name_cols = len(dataname)
    elif data.ndim == 2:
        rows, cols = data.shape
        name_cols = len(dataname)
        
    if cols != name_cols:
        raise ValueError('Data and dataname columns do no match %d and %d' % (cols,name_cols))
    
    secnum = mpc.SecFxp()
    
    test_sample = secnum(10)
    stype = type(test_sample)
    field = stype.field
    
    data_sec = np.vectorize(secnum)(data)
    print(data_sec.shape)
   
    m = len(hosts)
    t = 1
    N = rows
    #N = 3
    for k in np.arange(cols):
        for i in np.arange(N):
            # Generate shares for each data sample
            data_shares = [None]*m
            data_shares = thresha.random_split([data_sec[k, i].df], t, m)
                
            data_shares_str = []
            for other_pid, data in enumerate(data_shares):
                data = field.to_bytes(data)
                data_shares_str.append(base64.b64encode(data).decode())
                    
            #print(data_shares_str)
            timestamp = datetime.datetime.now().isoformat()
            for j in np.arange(m):
                sample = {
                    "timestamp" : timestamp,
                    dataname[k] : data_shares_str[j]
                }
                sample_json = json.dumps(sample)
                send_data(sample_json, datapart, hosts[j], ports[j])
                
 
def send_shares_mpc_combined(data, dataname, datapart, hosts, ports):
    #print(f'Sending combined data...')
        
    if isinstance(data, np.ndarray):
        data = data.flatten().tolist()
      
    secnum = mpc.SecFxp()
    test_sample = secnum(10)
    stype = type(test_sample)
    field = stype.field
    
    data_sec_ = np.vectorize(secnum)(data)
    data_sec_ = data_sec_.tolist()
    data_sec = [a.df for a in data_sec_]
   
    m = len(hosts)
    t = 1
    
    start = timer()
    # Generate shares for each data sample
    #data_shares = [None]*m
    data_shares = thresha.random_split(data_sec, t, m)
    end = timer()
    global running_time_compute_share
    running_time_compute_share = end - start
        
    data_shares_str = []
    for other_pid, data in enumerate(data_shares):
        data = field.to_bytes(data)
        data_shares_str.append(base64.b64encode(data).decode())
            
    #print(data_shares_str)
    start = timer()
    
    parallel = True
    if parallel:
        #print('send parallel')
        timestamp = datetime.datetime.now().isoformat()
        sample_json = []
        for j in np.arange(m):
            sample = {
                "timestamp" : timestamp,
                dataname[0] : data_shares_str[j]
            }
            sample_json.append(json.dumps(sample))
            #send_data(sample_json, datapart, hosts[j], ports[j])
        Parallel(n_jobs=m)(delayed(send_data)(sample_json[i], datapart, hosts[i], ports[i]) for i in np.arange(m))
    else:
        
        timestamp = datetime.datetime.now().isoformat()
        for j in np.arange(m):
            sample = {
                "timestamp" : timestamp,
                dataname[0] : data_shares_str[j]
            }
            sample_json = json.dumps(sample)
            send_data(sample_json, datapart, hosts[j], ports[j])
            
    global running_time_upload_share
    end = timer()
    running_time_upload_share = end - start
    
def send_shares_mpc_cleartext(data, dataname, datapart, hosts, ports):
    if isinstance(data, np.ndarray):
        data = data.tolist()
        
    data_str = ','.join(str(d) for d in data)
    
    timestamp = datetime.datetime.now().isoformat()
    
    data_str_encode = base64.b64encode(data_str.encode('utf-8')).decode()
    #print(f'data cleatext: {data_str_encode}')
    m = len(hosts)
    for j in np.arange(m):
        sample = {
                "timestamp" : timestamp,
                dataname[0] : data_str_encode
            }
        sample_json = json.dumps(sample)
        send_data(sample_json, datapart, hosts[j], ports[j])
                
def send_shares_mpc(data, dataname, datapart, hosts, ports, combined=False, privacy_scheme='secretshares'):
    hosts = [socket.gethostbyname(s) for s in hosts]
    
    if privacy_scheme == 'secretshares':
        if combined:
            send_shares_mpc_combined(data, dataname, datapart, hosts, ports)
        else:
            send_shares_mpc_single(data, dataname, datapart, hosts, ports)
    elif privacy_scheme == 'cleartext':
        send_shares_mpc_cleartext(data, dataname, datapart, hosts, ports)
    else:
        print('Unknown privacy scheme and data not sent...')

def request_mpc(function, host, port):
    print(f'request: {host}, {port}')
    r = requests.get(f'http://{host}:{port}/mpyc_compute?api={function}')
    return r
    
def mp_compute_functions_mpc(function, hosts, ports):
    m = len(hosts)
    results = None
    results = Parallel(n_jobs=m)(delayed(request_mpc)(function, hosts[m-i-1], ports[m-i-1]) for i in np.arange(m))
    #print(f'results: {results[0].text}')
    return results
     
def receive_computed_shares_mpc(function, hosts, ports):
    #start mpc servers
    results = mp_compute_functions_mpc(function, hosts, ports)
    print(f'response length: {len(results)}')
    
    secnum = mpc.SecFxp()
    data_sample = secnum(10)
    stype = type(data_sample)
    field = stype.field
    
    print(f'party 0: {results[0].text}')
    print(f'party 1: {results[1].text}')
    print(f'party 2: {results[2].text}')
    
    #combine shares
#    reconstructed_results = []
#    if results:
#        data_shares0 = [[field(x) for x in field.from_bytes(base64.b64decode(r + '='))] for r in results[0].text]
#        data_shares1 = [[field(x) for x in field.from_bytes(base64.b64decode(r + '='))] for r in results[1].text]
#        data_shares2 = [[field(x) for x in field.from_bytes(base64.b64decode(r + '='))] for r in results[2].text]
#        
#        data_shares = np.vstack((np.array(data_shares0), np.array(data_shares1), np.array(data_shares2)))
#        
#        for i in np.arange(data_shares.shape[1]):
#            points = [(j + 1, data_shares[j][i]) for j in np.arange(data_shares.shape[0])]
#            reconstructed_results.append(thresha.recombine(field, points))
#        
#    return reconstructed_results
#    
    