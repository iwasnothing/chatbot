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


    writeDic(input_token_index,'input_token_index')
    writeDic(target_token_index,'target_token_index')
    writeDic(reverse_input_char_index,'reverse_input_char_index')
    writeDic(reverse_target_char_index,'reverse_target_char_index')



    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text.split()):
            if char in input_token_index.keys():
                encoder_input_data[i, t, input_token_index[char]] = 1.
            else:
                encoder_input_data[i, t, input_token_index[pad_token]] = 1.
        for t, char in enumerate(target_text.split()):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            if char in target_token_index.keys():
                decoder_input_data[i, t, target_token_index[char]] = 1.
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.
            else:
                decoder_input_data[i, t, target_token_index[pad_token]] = 1.
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[pad_token]] = 1.


    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]


    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)


    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
    # Save model
    model.save(folder + 's2s.h5')


    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states


    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)


    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    encoder_model.save(folder + 's2s_enc.h5')
    decoder_model.save(folder + 's2s_dec.h5')


# MAIN

QList = []
AList = []
parent= 'data'
for d in os.listdir(parent):
    if os.path.isdir(parent+'/'+d) and  not os.path.exists(parent+'/'+d+'/' + 's2s.h5'):
        if os.path.exists(parent+'/'+d+'/' + 'QList.txt'):
            with open(parent+'/'+d+'/' + 'QList.txt','r') as f:
                QList = f.read()[:-1].split('\n')
        if os.path.exists(parent+'/'+d+'/' + 'AList.txt') :
            with open(parent+'/'+d+'/' + 'AList.txt','r') as f:
                AList = f.read()[:-1].split('\n')
        train(QList,AList,parent+'/'+d+'/')

