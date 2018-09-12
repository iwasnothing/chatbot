from __future__ import print_function
import tensorflow as tf
import pandas as pd
import os, sys
from keras.models import Model
from keras.layers import Input, LSTM, Dense,Embedding
from keras.utils import Sequence
from keras.callbacks import CSVLogger,ModelCheckpoint
import numpy as np
import json
import keras.backend as K

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, folder,for_training=True,shuffle=True):
        self.start_token = "START"
        self.end_token = "END"
        self.pad_token = "PAD"
        self.folder = folder
        self.shuffle = shuffle
        self.batch_size = 20  # Batch size for training.
        self.epochs = 100  # Number of epochs to train for.
        self.latent_dim = 300  # Latent dimensionality of the encoding space.
        self.QList = []
        self.AList = []
        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.input_characters.add(self.pad_token)
        self.input_characters.add(self.start_token)
        self.input_characters.add(self.end_token)
        self.target_characters = set()
        self.target_characters.add(self.pad_token)
        self.target_characters.add(self.start_token)
        self.target_characters.add(self.end_token)
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
        self.encoder_input_data = np.zeros( (self.batch_size, self.max_encoder_seq_length))
        self.decoder_input_data = np.zeros( (self.batch_size, self.max_decoder_seq_length))
        self.decoder_target_data = np.zeros( (self.batch_size,self.max_decoder_seq_length))
        self.for_training = for_training

        if os.path.exists(self.folder + 'QList.txt'):
            with open(self.folder + 'QList.txt','r') as f:
                self.QList = f.read()[:-1].split('\n')
        if os.path.exists(self.folder + 'AList.txt') :
            with open(self.folder + 'AList.txt','r') as f:
                self.AList = f.read()[:-1].split('\n')
        all_samples = min([len(self.QList), len(self.AList)])
        self.num_samples =  all_samples
        self.num_batches_per_epoch = int(self.num_samples/self.batch_size)
        self.train_num_batches_per_epoch = int( self.num_batches_per_epoch * 0.9 )
        self.verify_num_batches_per_epoch =  self.num_batches_per_epoch - self.train_num_batches_per_epoch
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
                    if self.input_count[char] > 0:
                        self.input_characters.add(char)
            for char in target_text.split():
                if char not in self.target_count.keys():
                    self.target_count[char] = 1
                else:
                    self.target_count[char] = self.target_count[char] + 1
                    if self.target_count[char] > 0:
                        self.target_characters.add(char)


        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])+2
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])+2


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
        #return self.num_batches_per_epoch
        if self.for_training == True:
            return self.train_num_batches_per_epoch
        else:
            return self.verify_num_batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if self.for_training == True:
            batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            batch_indexes = self.indexes[(self.train_num_batches_per_epoch+index)*self.batch_size:(self.train_num_batches_per_epoch+index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = batch_indexes

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
        #print(sz)
        self.encoder_input_data = np.zeros(
               (sz, self.max_encoder_seq_length),
                dtype='float32')
        self.decoder_input_data = np.zeros(
                (sz, self.max_decoder_seq_length),
                dtype='float32')
        self.decoder_target_data = np.zeros(
                (sz, self.max_decoder_seq_length,self.num_decoder_tokens),
                dtype='float32')
    
        for b,i in enumerate(list_IDs_temp):
            input_text = self.input_texts[i] 
            target_text = self.target_texts[i] 

            for t, char in enumerate(input_text.split()):
                if char in self.input_token_index.keys():
                    self.encoder_input_data[b, t] = self.input_token_index[char] 
            for t, char in enumerate(target_text.split()):
                if char in self.target_token_index.keys():
                    self.decoder_input_data[b, t] = self.target_token_index[char]
                    if t > 0:
                        self.decoder_target_data[b, t - 1, self.target_token_index[char] ] = 1.

        return [self.encoder_input_data,self.decoder_input_data], self.decoder_target_data

    def real_loss(self,y_true, y_pred):
        y_true_flatten = K.flatten(y_true)
        num_total_elements = K.sum(y_true_flatten)
        y_pred_flatten = K.flatten(y_pred)
        y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
        # cross_entropy = K.dot(y_true_flatten, K.transpose(y_pred_flatten_log))
        cross_entropy = tf.reduce_sum(tf.multiply(y_true_flatten, y_pred_flatten_log))
        mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())
        return mean_cross_entropy


def train(folder):

    training_generator = DataGenerator(folder,True,True)
    validation_generator = DataGenerator(folder,False,True)

    g = training_generator
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    # Define an input sequence and process it.
    #encoder_inputs = Input(shape=(None, g.num_encoder_tokens))
    encoder_inputs = Input(shape=(g.max_encoder_seq_length,),dtype='int32')
    encoder_embed = Embedding(g.num_encoder_tokens,g.latent_dim, input_length=g.max_encoder_seq_length)
    encoder = LSTM(g.latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_embed(encoder_inputs))
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    #decoder_inputs = Input(shape=(None, g.num_decoder_tokens))
    decoder_inputs = Input(shape=(g.max_decoder_seq_length,), dtype='int32')
    decoder_embed = Embedding(g.num_decoder_tokens,g.latent_dim, input_length=g.max_decoder_seq_length)
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(g.latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embed(decoder_inputs),
                                     initial_state=encoder_states)
    decoder_dense = Dense(g.num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)


    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


    # Run training
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.compile(optimizer='rmsprop', loss=training_generator.real_loss)
    # Generators

    callback = [CSVLogger(filename=folder+'trainlog.csv',append=True),
                ModelCheckpoint(folder + 'model_check_{epoch:02d}.h5',
                                save_best_only=True,
                                save_weights_only=False)]

    #model.fit_generator(train_batches)
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=training_generator.num_batches_per_epoch, epochs=20,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.num_batches_per_epoch,
                    callbacks = callback,
                    use_multiprocessing=True,
                    workers=1)
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


    decoder2_state_input_h = Input(shape=(g.latent_dim,))
    decoder2_state_input_c = Input(shape=(g.latent_dim,))
    decoder2_states_inputs = [decoder2_state_input_h, decoder2_state_input_c]
    decoder2_outputs, state2_h, state2_c = decoder_lstm(
        decoder_embed(decoder_inputs), initial_state=decoder2_states_inputs)
    decoder2_states = [state2_h, state2_c]
    decoder2_outputs = decoder_dense(decoder2_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder2_states_inputs,
        [decoder2_outputs] + decoder2_states)
    
    encoder_model.save(folder + 's2s_enc.h5')
    decoder_model.save(folder + 's2s_dec.h5')
    


# MAIN
if __name__ == '__main__':
    parent= 'data'
    for d in os.listdir(parent):
        if os.path.isdir(parent+'/'+d) and  not os.path.exists(parent+'/'+d+'/' + 's2s.h5'):
            train(parent+'/'+d+'/')

