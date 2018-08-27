from __future__ import print_function
import pandas as pd
import os, sys
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

import json

start_token = "START "
end_token = " END"
pad_token = " PAD "

parent = 'data'
flist = []
for d in os.listdir(parent):
    if os.path.isdir(parent+'/'+d) and  os.path.exists(parent+'/'+d+'/' + 's2s.h5'):
        if 'T' in d:
            flist.append(d)
folder = flist[np.random.choice(len(flist),1)[0]]
try:
    if os.path.exists(parent + '/' + folder + '/' + 'count.csv'):
        with open(parent + '/' + folder + '/' + 'count.csv', 'r') as f:
            lines = f.read()[:-1].split('\n')
            if len(lines) >= 5:
                num_samples = int(lines[0].split(':')[1])        
                num_encoder_tokens = int(lines[1].split(':')[1])        
                num_decoder_tokens = int(lines[2].split(':')[1])        
                max_encoder_seq_length = int(lines[3].split(':')[1])        
                max_decoder_seq_length = int(lines[4].split(':')[1])        
    
    with open(parent + '/' + folder + '/' + 'input_token_index.json', 'r') as f:
            input_token_index = json.load(f)
    with open(parent + '/' + folder + '/' + 'target_token_index.json', 'r') as f:
            target_token_index = json.load(f)
        
    with open(parent + '/' + folder + '/' + 'reverse_input_char_index.json', 'r') as f:
            reverse_input_char_index = json.load(f)
    with open(parent + '/' + folder + '/' + 'reverse_target_char_index.json', 'r') as f:
            reverse_target_char_index = json.load(f)

except OSError as err:
    print("cannot read count")
    print(err)
   
def decode_sequence(input_texts):

    encoder_model = load_model(parent + '/' + folder + '/' + 's2s_enc.h5')
    decoder_model = load_model(parent + '/' + folder + '/' + 's2s_dec.h5')

    num_encoder_tokens = encoder_model.get_input_shape_at(0)[2]
    num_decoder_tokens = decoder_model.get_output_shape_at(0)[0][2]
    max_encoder_seq_length = 12
    max_decoder_seq_length = 12

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text.split()):
            if char in input_token_index.keys():
                encoder_input_data[i, t, input_token_index[char]] = 1.
            else:
                encoder_input_data[i, t, input_token_index[pad_token]] = 1.




    # Encode the input as state vectors.
    states_value = encoder_model.predict(encoder_input_data)


    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['START']] = 1.


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
        sampled_char = reverse_target_char_index[str(sampled_token_index)]
        decoded_sentence += sampled_char + " "


        # Exit condition: either hit max length
        # or find stop character.
        if ('START' in sampled_char or 'END' in sampled_char or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True


        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.


        # Update states
        states_value = [h, c]


    return decoded_sentence




    # Take one sequence (part of the training set)
    # for trying out decoding.
with open('input_texts.txt','r') as f:
    input_seq = f.read()[:-1].split('\n')
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_seq)
    print('Decoded sentence:', decoded_sentence)
