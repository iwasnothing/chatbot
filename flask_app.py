
# A very simple Flask Hello World app for you to get started with...

from flask import Flask
from flask import Flask, request
from flask import Response
from flask_cors import CORS, cross_origin
import telepot
import urllib3
import json
import requests
#from mydecoder import MyDecoder

#app = Flask(__name__)

#@app.route('/chat')
#def hello_world():
    #return 'Hello from Flask!'



proxy_url = "http://proxy.server:3128"
telepot.api._pools = {
    'default': urllib3.ProxyManager(proxy_url=proxy_url, num_pools=3, maxsize=10, retries=False, timeout=30),
}
telepot.api._onetime_pool_spec = (urllib3.ProxyManager, dict(proxy_url=proxy_url, num_pools=1, maxsize=1, retries=False, timeout=30))

secret = "bot94077079"
bot = telepot.Bot('644105824:AAEHoj9Lhf0P_3Iv7LFNgOZAfI2t044UTW4')
bot.setWebhook("https://iwasnothing.pythonanywhere.com/{}".format(secret), max_connections=1)

app = Flask(__name__)
#dc = MyDecoder()

@app.route('/{}'.format(secret), methods=["POST"])
def telegram_webhook():
    update = request.get_json()
    if "message" in update:
        chat_id = update["message"]["chat"]["id"]
        jstr = json.dumps(update["message"])
        d = json.loads(jstr)
        print(d)
        t = d["text"]
        url = 'http://35.239.154.82:8080'
        r = requests.post(url, data=json.dumps(update["message"]))
        #r = dc.decode_sequence([t])
        print(t)
        bot.sendMessage(chat_id, ".....{}".format(r.text))
    return "OK"

cors = CORS(app, resources={r"/chatbot": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'



@app.route('/chat')
def hello_world():
    return 'Hello from Flask!'

@app.route('/chatbot', methods = ['POST','OPTIONS'])
@cross_origin(origin='*',headers=['Content-Type','Authorization', 'Access-Control-Allow-Headers', 'X-Requested-With', 'Origin', 'Accept' ])
def mybot():
    url = 'http://35.239.154.82:8080'
    r = requests.post(url, data=request.data)
    #resp = Response(r.text, status=200, mimetype='application/x-www-form-urlencoded; charset=UTF-8')

    #resp.headers['Access-Control-Allow-Origin'] = "*"
    #resp.headers['Access-Control-Allow-Methods'] = "GET, POST, OPTIONS"
    #resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Access-Control-Allow-Headers,
    #Authorization, X-Requested-With, Origin, Accept"
    #resp.headers['Content-type'] = 'application/x-www-form-urlencoded; charset=UTF-8'
    return r.text