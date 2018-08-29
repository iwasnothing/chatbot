from __future__ import print_function
import pandas as pd
import os, sys
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import json


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, folder,for_training=True,shuffle=True):
        self.start_token = "START "
        self.end_token = " END"
        self.pad_token = " PAD "
        self.folder = folder
        self.shuffle = shuffle
        self.batch_size = 20  # Batch size for training.
        self.epochs = 100  # Number of epochs to train for.
        self.latent_dim = 16  # Latent dimensionality of the encoding space.
        self.QList = []
        self.AList = []
        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.input_characters.add(self.pad_token)
        self.target_characters = set()
        self.target_characters.add(self.pad_token)
        self.input_count = {}
        self.target_count = {}
        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0
        self.max_encoder_seq_length = 0
        self.max_decoder_seq_length = 0
        self.num_samples = 0
        self.input_token_index = {}
        self.target_token_index = {}
        self.reverse_input_char_index = {}
        self.reverse_target_char_index = {}
        self.encoder_input_data = np.zeros( (self.batch_size, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        self.decoder_input_data = np.zeros( (self.batch_size, max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        self.decoder_target_data = np.zeros( (self.batch_size, max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        self.for_training = for_training

        if os.path.exists(self.folder + 'QList.txt'):
            with open(self.folder + 'QList.txt','r') as f:
                self.QList = f.read()[:-1].split('\n')
        if os.path.exists(self.folder + 'AList.txt') :
            with open(self.folder + 'AList.txt','r') as f:
                self.AList = f.read()[:-1].split('\n')
        all_samples = min([len(self.QList), len(self.AList)])
        self.num_samples =  all_samples
        self.num_batches_per_epoch = int((self.num_samples - 1) / self.batch_size) + 1
        self.indexes = np.arange(self.num_samples)

        for i in range(self.num_samples):
            input_text = self.QList[i]
            target_text = self.AList[i] 
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text.split():
                if char not in self.input_count.keys():
                    self.input_count[char] = 1
                else:
                    self.input_count[char] = self.input_count[char] + 1
                    if self.input_count[char] > 2:
                        self.input_characters.add(char)
            for char in target_text.split():
                if char not in target_count.keys():
                    self.target_count[char] = 1
                else:
                    self.target_count[char] = self.target_count[char] + 1
                    if self.target_count[char] > 2:
                        self.target_characters.add(char)


        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])


        print('Number of samples:', len(self.input_texts))
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)

        with open(self.folder + 'count.csv', 'w') as the_file:
            print('Number of samples:', len(self.input_texts),file=the_file)
            print('Number of unique input tokens:', self.num_encoder_tokens,file=the_file)
            print('Number of unique output tokens:', self.num_decoder_tokens,file=the_file)
            print('Max sequence length for inputs:', self.max_encoder_seq_length,file=the_file)
            print('Max sequence length for outputs:', self.max_decoder_seq_length,file=the_file)

        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(self.target_characters)])

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())


        self.writeDic(self.input_token_index,self.folder + 'input_token_index')
        self.writeDic(self.target_token_index,self.folder + 'target_token_index')
        self.writeDic(self.reverse_input_char_index,self.folder + 'reverse_input_char_index')
        self.writeDic(self.reverse_target_char_index,self.folder + 'reverse_target_char_index')

    def writeDic(self,data,name):
        with open(name + '.json', 'w') as outfile:
            json.dump(data, outfile)

    def __len__(self):
        return self.num_batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        sz = len(list_IDs_temp)

        self.encoder_input_data = np.zeros(
               (sz, self.max_encoder_seq_length, self.num_encoder_tokens),
                dtype='float32')
        self.decoder_input_data = np.zeros(
                (sz, self.max_decoder_seq_length, self.num_decoder_tokens),
                dtype='float32')
        self.decoder_target_data = np.zeros(
                (sz, self.max_decoder_seq_length, self.num_decoder_tokens),
                dtype='float32')
    
        for i in list_IDs_temp:
            input_text = self.input_texts[i]
            target_text = self.target_texts[i]
            for t, char in enumerate(input_text.split()):
                if char in self.input_token_index.keys():
                    self.encoder_input_data[i, t, self.input_token_index[char]] = 1.
                else:
                    self.encoder_input_data[i, t, self.input_token_index[pad_token]] = 1.
            for t, char in enumerate(target_text.split()):
                    # decoder_target_data is ahead of decoder_input_data by one timestep
                if char in self.target_token_index.keys():
                    self.decoder_input_data[i, t, self.target_token_index[char]] = 1.
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    if t > 0:
                        self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.
                else:
                    self.decoder_input_data[i, t, self.target_token_index[pad_token]] = 1.
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    if t > 0:
                        self.decoder_target_data[i, t - 1, self.target_token_index[pad_token]] = 1.

        return [encoder_input_data,decoder_input_data], decoder_target_data



def train(folder):

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
    # Generators
    training_generator = DataGenerator(folder,True)
    validation_generator = DataGenerator(folder,False)
    #model.fit_generator(train_batches)
    model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
    #model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          #batch_size=batch_size,
          #epochs=epochs,
          #validation_split=0.2)
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

parent= 'data'
for d in os.listdir(parent):
    if os.path.isdir(parent+'/'+d) and  not os.path.exists(parent+'/'+d+'/' + 's2s.h5'):
        train(parent+'/'+d+'/')

