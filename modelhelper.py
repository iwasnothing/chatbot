import tensorflow as tf
import pandas as pd
import os, sys
from keras.models import Model
from keras.layers import Input, LSTM, Dense,Embedding
from keras.utils import Sequence
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
        self.latent_dim = 16  # Latent dimensionality of the encoding space.
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

