from __future__ import print_function
import pandas as pd
import os, sys
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import json

start_token = "START "
end_token = " END"
pad_token = " PAD "



def writeDic(data,name):
    with open(name + '.json', 'w') as outfile:
        json.dump(data, outfile)

def train(QList,AList,folder):
    batch_size = 20  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 16  # Latent dimensionality of the encoding space.
    input_texts = []
    target_texts = []
    input_characters = set()
    input_characters.add(pad_token)
    target_characters = set()
    target_characters.add(pad_token)
    input_count = {}
    target_count = {}

    min_samples = min([len(QList), len(AList)])
    num_samples = min_samples
    #rclist = np.random.choice(min_samples,num_samples,replace=False)
    for i in range(num_samples):
    #for i in rclist:
        input_text = QList[i]
        target_text = AList[i] 
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text.split():
            if char not in input_count.keys():
                input_count[char] = 1
            else:
                input_count[char] = input_count[char] + 1
                if input_count[char] > 2:
                    input_characters.add(char)
        for char in target_text.split():
            if char not in target_count.keys():
                target_count[char] = 1
            else:
                target_count[char] = target_count[char] + 1
                if target_count[char] > 2:
                    target_characters.add(char)


    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])


    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    with open(folder + 'count.csv', 'w') as the_file:
        print('Number of samples:', len(input_texts),file=the_file)
        print('Number of unique input tokens:', num_encoder_tokens,file=the_file)
        print('Number of unique output tokens:', num_decoder_tokens,file=the_file)
        print('Max sequence length for inputs:', max_encoder_seq_length,file=the_file)
        print('Max sequence length for outputs:', max_decoder_seq_length,file=the_file)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())


    writeDic(input_token_index,folder + 'input_token_index')
    writeDic(target_token_index,folder +'target_token_index')
    writeDic(reverse_input_char_index,folder + 'reverse_input_char_index')
    writeDic(reverse_target_char_index,folder + 'reverse_target_char_index')




# MAIN

QList = []
AList = []
parent= 'data'
for d in os.listdir(parent):
#for d in ['P538']:
    if os.path.isdir(parent+'/'+d):
        if os.path.exists(parent+'/'+d+'/' + 'QList.txt'):
            with open(parent+'/'+d+'/' + 'QList.txt','r') as f:
                QList = f.read()[:-1].split('\n')
        if os.path.exists(parent+'/'+d+'/' + 'AList.txt') :
            with open(parent+'/'+d+'/' + 'AList.txt','r') as f:
                AList = f.read()[:-1].split('\n')
        train(QList,AList,parent+'/'+d+'/')

