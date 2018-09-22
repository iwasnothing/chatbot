
# A very simple Flask Hello World app for you to get started with...

from flask import Flask
from flask import Flask, request
from flask import Response
from flask_cors import CORS, cross_origin
import urllib3
import json
import requests
from mydecoder import MyDecoder


app = Flask(__name__)
dc = MyDecoder('data/P63/')

cors = CORS(app, resources={r"/chatbot": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'



@app.route('/chat')
def hello_world():
    return 'Hello from Flask!'

@app.route('/chatbot', methods = ['POST','OPTIONS'])
@cross_origin(origin='*',headers=['Content-Type','Authorization', 'Access-Control-Allow-Headers', 'X-Requested-With', 'Origin', 'Accept' ])
def mybot():
    d = json.loads(request.data)
    t = d['text']
    
    r = dc.decode_sequence([t])

    return r
