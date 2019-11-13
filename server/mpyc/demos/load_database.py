import os
import base64

from mpyc.runtime import mpc
from pymongo import MongoClient

def load_data(data, datapart, limitN=0, data_secrecy='secretshares'):

    # MongoDB client
    Party = os.getenv(f"Party")
    print(f'Loading data for party: {Party}...')

    mongo_url = "mongodb://mongo:27017"
    #print(f'mongo url: {mongo_url}')
    client = MongoClient(mongo_url)
    db = client.test_db

    # Needed to convert shares in string format to secret number format (can be moved to api scripts)
    secnum = mpc.SecFxp()
    data_sample = secnum(10)
    field = type(data_sample).field

    # load data from database
    samples = None
    #cursor = db.sensor_data.find()
    if datapart == 'train':
        sensor_data = db.train.sensor_data
    elif datapart == 'test':
        sensor_data = db.test.sensor_data
    elif datapart == 'model':
        sensor_data = db.model.sensor_data
    cursor = sensor_data.find().limit(limitN)
    print("cursor", cursor)
    samples = [s[data] for s in cursor]

    if samples is None:
        return None
    if data_secrecy == 'secretshares':
       #convert from base64 string to secure number
       data_shares = [[secnum(field(x)) for x in field.from_bytes(base64.b64decode(r))] for r in samples]
       return data_shares
    elif data_secrecy == 'cleartext':
        data_shares = [[secnum(float(x)) for x in base64.b64decode(r).decode().split(',')] for r in samples]
        #print(f'data shares: {dir(data_shares[0])}')
        return data_shares
    else:
        return None
