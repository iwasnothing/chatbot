from keras.models import load_model
from keras.models import Model
from keras.layers import Input, LSTM, Dense,Embedding,RepeatVector, Flatten, Reshape
import tensorflow as tf
import numpy as np
import json
import keras.backend as K
import os, sys

"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""

def real_loss(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    num_total_elements = K.sum(y_true_flatten)
    y_pred_flatten = K.flatten(y_pred)
    y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
    # cross_entropy = K.dot(y_true_flatten, K.transpose(y_pred_flatten_log))
    cross_entropy = tf.reduce_sum(tf.multiply(y_true_flatten, y_pred_flatten_log))
    mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())
    return mean_cross_entropy


class MyDecoder:
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

        
    def writeDic(self,data,name):
        with open(name + '.json', 'w') as outfile:
            json.dump(data, outfile)
            
    def __init__(self, folder):
        self.start_token = "START"
        self.end_token = "END"
        self.pad_token = "PAD"
        self.folder = folder
        #self.shuffle = shuffle
        self.batch_size = 20  # Batch size for training.
        self.epochs = 20  # Number of epochs to train for.
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
        #self.for_training = for_training

        if os.path.exists(self.folder + 'QList.txt'):
            with open(self.folder + 'QList.txt','r') as f:
                self.QList = f.read()[:-1].split('\n')
        if os.path.exists(self.folder + 'AList.txt') :
            with open(self.folder + 'AList.txt','r') as f:
                self.AList = f.read()[:-1].split('\n')
        all_samples = min([len(self.QList), len(self.AList)])
        self.num_samples =  0


        for i in range(all_samples):
            input_text = self.QList[i]
            target_text = self.AList[i]
            tokens = target_text.split()
            target_len = len(tokens)
            for j in range( target_len ):
                if j > 0:
                    self.input_texts.append(input_text)
                    partial_target = " ".join(tokens[0:j])
                    self.target_texts.append(partial_target)
                    self.num_samples = self.num_samples + 1
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

        self.num_batches_per_epoch = int(self.num_samples/self.batch_size)
        self.train_num_batches_per_epoch = int( self.num_batches_per_epoch * 0.9 )
        self.verify_num_batches_per_epoch =  self.num_batches_per_epoch - self.train_num_batches_per_epoch
        self.indexes = np.arange(self.num_samples)
        
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])


        print('Number of samples:', len(self.input_texts))
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)


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



    def decode_sequence(self,input_texts):


        encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length),
            dtype='float32')
        for i, input_text in enumerate(input_texts):
            print(input_text)
            input_text = self.start_token + " " + input_text+ " " + self.end_token

            for t, char in enumerate(input_text.split()):
                if char in self.input_token_index.keys():
                    encoder_input_data[i, t] = self.input_token_index[char]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        K.clear_session()

        encoder_inputs = Input(shape=(self.max_encoder_seq_length,),dtype='int32')
        encoder_embed = Embedding(self.num_encoder_tokens,self.latent_dim, input_length=self.max_encoder_seq_length)
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_embed(encoder_inputs))
        z1 = tf.random_normal(tf.shape(state_h))
        z2 = tf.random_normal(tf.shape(state_c))
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h + z1, state_c + z2]

        # Set up the decoder, using `encoder_states` as initial state.
        #decoder_inputs = Input(shape=(None, g.num_decoder_tokens))
        decoder_inputs = Input(shape=(self.max_decoder_seq_length,), dtype='int32')
        decoder_embed = Embedding(self.num_decoder_tokens,self.latent_dim, input_length=self.max_decoder_seq_length)
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embed(decoder_inputs),
                                         initial_state=encoder_states)
        #decoder_flatten = Flatten()
        #decoder_outputs = decoder_flatten(decoder_outputs)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        #decoder_reshape = Reshape((g.max_decoder_seq_length,g.num_decoder_tokens))
        #decoder_outputs = decoder_reshape(decoder_outputs )


        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        #model = Model(encoder_inputs, decoder_outputs)
        
        # Run training
        #model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.model.compile(optimizer='rmsprop', loss=real_loss)

        self.model.load_weights(self.folder + 's2s.h5')

        # Generate empty target sequence of length 1.
        target_seq = np.zeros(
                (1, self.max_decoder_seq_length),
                dtype='float32')
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0 ] = self.target_token_index[self.start_token]
        #decoder_input_data = [target_seq]

        #states_value = self.encoder_model.predict(encoder_input_data)
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        predicted_count=0
        while not stop_condition:
            output_tokens = self.model.predict([encoder_input_data,target_seq])
            #decoder_input_data = [target_seq]
            #output_tokens = model.predict([encoder_input_data,target_seq])
            # Sample a token
            predicted_count = predicted_count + 1
            pdf = output_tokens[0, predicted_count, :]
            sampled_token_index = np.random.choice(len(pdf),p=pdf)
            #sampled_token_index = np.argmax(pdf)
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char + " "


            # Exit condition: either hit max length
            # or find stop character.
            if ('END' in sampled_char or sampled_char == '?' or sampled_char == '.' or
               len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True


            # Update the target sequence (of length 1).
            target_seq[0, predicted_count] = sampled_token_index


            # Update states
            #states_value = [h, c]


        return decoded_sentence


