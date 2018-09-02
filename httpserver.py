#!/usr/bin/env python3
from __future__ import print_function
from cgi import parse_header, parse_multipart
from urllib.parse import parse_qs
import pandas as pd
import os, sys
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import json
from sys import argv
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
from urllib.parse import parse_qs

"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""



class S(BaseHTTPRequestHandler):
    start_token = "START"
    end_token = "END"
    pad_token = "PAD"
    max_encoder_seq_length = 10
    max_decoder_seq_length = 10
    num_encoder_tokens = 30
    num_decoder_tokens = 30

    def score(self,sentence,d):
        count = 0    
        with open(self.parent + '/' + d + '/' + 'input_token_index.json', 'r') as f:
            self.input_token_index = json.load(f)
            for key, value in self.input_token_index.items():
                if key in sentence:
                    count = count + 1
        return count
    def myinit(self,sentence):
        self.parent = 'data'
        self.flist = []
        max_count = 0
        for d in os.listdir(self.parent):
            if os.path.isdir(self.parent+'/'+d) and  os.path.exists(self.parent+'/'+d+'/' + 's2s.h5'):
                count = self.score(sentence,d)
                if count >= max_count:
                    max_count = count
                    self.folder = d
        #self.folder = self.flist[np.random.choice(len(self.flist),1)[0]]
        try:
            if os.path.exists(self.parent + '/' + self.folder + '/' + 'count.csv'):
                with open(self.parent + '/' + self.folder + '/' + 'count.csv', 'r') as f:
                    lines = f.read()[:-1].split('\n')
                    if len(lines) >= 5:
                        num_samples = int(lines[0].split(':')[1])        
                        self.num_encoder_tokens = int(lines[1].split(':')[1])        
                        self.num_decoder_tokens = int(lines[2].split(':')[1])        
                        self.max_encoder_seq_length = int(lines[3].split(':')[1])        
                        self.max_decoder_seq_length = int(lines[4].split(':')[1])        
    
            with open(self.parent + '/' + self.folder + '/' + 'input_token_index.json', 'r') as f:
                self.input_token_index = json.load(f)
            with open(self.parent + '/' + self.folder + '/' + 'target_token_index.json', 'r') as f:
                self.target_token_index = json.load(f)
            
            with open(self.parent + '/' + self.folder + '/' + 'reverse_input_char_index.json', 'r') as f:
                self.reverse_input_char_index = json.load(f)
            with open(self.parent + '/' + self.folder + '/' + 'reverse_target_char_index.json', 'r') as f:
                self.reverse_target_char_index = json.load(f)

        except OSError as err:
            print("cannot read count")
            print(err)
   

    def _set_response(self):
        self.send_response(200, "ok")       
        self.send_header('Access-Control-Allow-Origin', '*')                
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With, Origin, Accept") 
        self.send_header('Content-type', 'application/x-www-form-urlencoded; charset=UTF-8')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_OPTIONS(self):           
        self._set_response()

    def do_POST(self):
        print("start POST")
        p = (self.rfile.read(int(self.headers['Content-Length']))).decode("utf-8")
        print(p)
        line = json.loads(p)['text']
        input_seq = []
        input_seq.append(line)
        self.myinit(line)
        decoded_sentence = self.decode_sequence(input_seq)
        print('-')
        print('Input sentence:', input_seq)
        print('Decoded sentence:', decoded_sentence)

        self._set_response()
        self.wfile.write(decoded_sentence.format(self.path).encode('utf-8'))

    def decode_sequence(self,input_texts):

        encoder_model = load_model(self.parent + '/' + self.folder + '/' + 's2s_enc.h5')
        decoder_model = load_model(self.parent + '/' + self.folder + '/' + 's2s_dec.h5')
        print(self.folder)
        self.num_encoder_tokens = encoder_model.get_input_shape_at(0)[2]
        self.num_decoder_tokens = decoder_model.get_output_shape_at(0)[0][2]

        encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')
        for i, input_text in enumerate(input_texts):
            for t, char in enumerate(input_text.split()):
                if char in self.input_token_index.keys():
                    encoder_input_data[i, t, self.input_token_index[char]] = 1.
                else:
                    print(i)
                    print(t)
                    print(self.input_token_index[self.end_token])
                    encoder_input_data[i, t, self.input_token_index[self.end_token]] = 1.




        # Encode the input as state vectors.
        states_value = encoder_model.predict(encoder_input_data)


        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index['START']] = 1.


        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)


            # Sample a token
            pdf = output_tokens[0, -1, :]
            sampled_token_index = np.random.choice(len(pdf),p=pdf)
            #print("sample " + str(sampled_token_index) + " out of " + str(len(reverse_target_char_index) ) )
            sampled_char = self.reverse_target_char_index[str(sampled_token_index)]
            decoded_sentence += sampled_char + " "


            # Exit condition: either hit max length
            # or find stop character.
            if ('START' in sampled_char or 'END' in sampled_char or
               len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True


            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.


            # Update states
            states_value = [h, c]


        return decoded_sentence

def run(server_class=HTTPServer, handler_class=S, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')




if __name__ == '__main__':
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()

