import mpyc
import json
import sys
import os
import logging
import requests
import time
import datetime

import subprocess
from flask import abort, Flask, render_template, request
from pymongo import MongoClient

app = Flask(__name__,static_folder='./static')
logging.basicConfig(level=logging.DEBUG)
l = logging.getLogger("web_app")

# MongoDB client
#client = MongoClient("mongodb://172.20.96.113:27017")
client = MongoClient("mongodb://mongo:27017")
db = client.test_db
main_wd = os.getcwd()
is_running = False

def store_sample(line, datapart):
    """
    Stores a single data sample within MongoDB.

    Input:
        line : line as a dictionary.
    Output:
        success : bool if DB insertion was successful.
    """
    try:
        if datapart == 'train':
            sensor_data = db.train.sensor_data
        elif datapart == 'test':
            sensor_data = db.test.sensor_data
        elif datapart == 'model':
            sensor_data = db.model.sensor_data
        sensor_id = sensor_data.insert_one(line).inserted_id
        l.debug(f"Inserted line with id {sensor_id}")
    except Exception as e:
        l.error(e)
        return False
    return True


# Data store API
@app.route("/store", methods=["PUT"])
def put_training_data():
    datapart = request.args.get('datapart')
    if not request.is_json:
        l.error("Request was not JSON")
        return abort(400)
    l.info(f"/store received data: {request.json}")
    line = json.loads(request.json)
    store_sample(line, datapart)
    return ""


# Clears all database entries
@app.route("/clear_all")
def clear_all():
    datapart = request.args.get('datapart')
    if datapart == 'train':
        sensor_data = db.train.sensor_data
    elif datapart == 'test':
        sensor_data = db.test.sensor_data
    elif datapart == 'model':
        sensor_data = db.model.sensor_data
    try:
        sensor_data.drop()
    except Exception as e:
        l.error(e)
        return "Failed"
    return "Success"


@app.route("/fetch")
def fetch_all():
    datapart = request.args.get('datapart')
    if datapart == 'train':
        sensor_data = db.train.sensor_data
    elif datapart == 'test':
        sensor_data = db.test.sensor_data
    elif datapart == 'model':
        sensor_data = db.model.sensor_data
    cursor = sensor_data.find()
    samples = [s for s in cursor]
    return render_template("fetch_all.html", items=samples)


@app.route("/")
def index():
    return render_template("demo.html")


@app.route("/mpyc_launch", methods=["GET"])
def mpyc_launch():

    def get_api_name(api_name):
        return api_name + '.py'

    http_arg = request.args.get('api')
    l.debug(f'Arg api : {http_arg}')
    script_name = get_api_name(http_arg)
    l.debug(f'Api : {script_name}')
    if script_name is None:
        return "400"

    l.debug(request)
    os.chdir(main_wd)
    test_path="./mpyc/demos"
    os.chdir(test_path)
    # Raise other parties
    Party = os.getenv(f"Party")
    l.debug(f'Party: {Party}')
    if Party == '0':
        global is_running
        if is_running:
            return '200'
        else:
            is_running = True
        for i in range(int(os.getenv('N_PARTIES')) - 1, 0, -1):
            party_host = os.getenv(f'PARTY_{i}_HOST')
            party_port = os.getenv(f'PARTY_{i}_PORT')
            host_addr = f'http://{party_host}:{party_port}/mpyc_launch?api={http_arg}'
            l.debug(f'Target host: {host_addr}')
            r = requests.get(host_addr)
            l.debug(f'Party {i} response: {r.text}')
            time.sleep(2.50)
        # Fetch and parse result
        l.debug(f'{datetime.datetime.now()}: start mpyc script...')
        #raw_result = os.popen(f"python -u run.py 3 average.py").read()
        process = subprocess.Popen(['python', script_name, '-c', f'party{3}_0.ini'], stdout=DEVNULL)#stdout=subprocess.PIPE)#shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        stdout, stderr = process.communicate()
        is_running = False
        l.debug(f'{datetime.datetime.now()}: end mpyc compute script...')

        #print(f'output: {stdout.decode().split()}')
        output_formatted = stdout.decode().split('$$$')[1]
        output_formatted = output_formatted.split('$$$')[0]
        output_formatted = output_formatted.strip()
        output_formatted = output_formatted.replace('\n', '')
        output_formatted = output_formatted.strip(',')
        l.debug(f'{output_formatted}')
        #for s in output_formatted.split(','):
        #    l.debug(f'{s}')
        #output = [float(s) for s in output_formatted.split(',')]
        #return render_template("demo.html", result=output)
        #return render_template("fetch_all.html", items=output)
        return output_formatted

    else:
        l.debug(f'Party config: party{3}_{Party}.ini')
        #os.popen(f"python average.py -c party{3}_{Party}.ini > /dev/null 2>&1")
        os.system(f'python {script_name} -c party{3}_{Party}.ini &')
        return "200"


@app.route("/mpyc_compute", methods=["GET"])
def mpyc_compute():

    def get_api_name(api_name):
        return api_name + '.py'

    http_arg = request.args.get('api')
    l.debug(f'Compute arg api : {http_arg}')
    script_name = get_api_name(http_arg)
    l.debug(f'Compute api : {script_name}')
    if script_name is None:
        return "400"

    #l.debug(request)
    os.chdir(main_wd)
    test_path="./mpyc/demos"
    os.chdir(test_path)
    # Raise other parties
    Party = os.getenv(f"Party")
    l.debug(f'Party: {Party}')

    if Party == 0 or Party == '0':
        time.sleep(2*0.50)
    if Party == 1 or Party == '1':
        time.sleep(0.50)

    l.debug(f'{datetime.datetime.now()}: start mpyc compute script...')
    #raw_result = os.popen(f"python -u run.py 3 average.py").read()
    process = subprocess.Popen(['python', script_name, '-c', f'party{3}_{Party}.ini'], stdout=subprocess.PIPE)#shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    stdout, stderr = process.communicate()

    output_formatted = stdout.decode().split('$$$')[1]
    output_formatted = output_formatted.split('$$$')[0]
    output_formatted = output_formatted.strip()
    output_formatted = output_formatted.replace('\n', '')
    output_formatted = output_formatted.strip(',')
    l.debug(f'{output_formatted}')
    return output_formatted


# Inference API
@app.route("/test", methods=["GET"])
def get_inference():
    # Not yet implemented
    return abort(400)


if __name__ == '__main__':
    l.debug("Initialising server...")
    Party = os.getenv(f"Party")
    host_name = f'server{Party}_web_1'
    print(f'host name: {host_name}')
    app.run(debug=False,
            #ssl_context=(os.getenv("CERT_PATH"), os.getenv("SECRET_PATH")),
            host=host_name)
