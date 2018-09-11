from keras.models import load_model
from keras.models import Model
from keras.layers import Input, LSTM, Dense
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
    def __init__(self):
        self.myinit('START')
        self.initmodel()

    def myinit(self,sentence):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        self.parent = '/home/iwasnothing/emb2/chatbot/data'
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

    def initmodel(self):
        self.latent_dim = 100

        #encoder_model = load_model(self.parent + '/' + self.folder + '/' + 's2s_enc.h5')
        #decoder_model = load_model(self.parent + '/' + self.folder + '/' + 's2s_dec.h5')
        #model = load_model(self.parent + '/' + self.folder + '/' + 's2s.h5')
        self.model = load_model(self.parent + '/' + self.folder + '/' + 's2s.h5', custom_objects={'real_loss': real_loss})
        self.model.summary()
        encoder_inputs = self.model.input[0]   # input_1
        encoder_embed = self.model.layers[2]   # input_1
        encoder_outputs, state_h_enc, state_c_enc = self.model.layers[4].output   # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = self.model.input[1]   # input_2
        decoder_embed = self.model.layers[3]   # input_2
        decoder_state_input_h = Input(shape=(self.latent_dim,), name='input_3')
        decoder_state_input_c = Input(shape=(self.latent_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = self.model.layers[5]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_embed(decoder_inputs), initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = self.model.layers[6]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    def decode_sequence(self,input_texts):


        self.max_encoder_seq_length = self.encoder_model.get_input_shape_at(0)[1]
        self.max_decoder_seq_length = self.decoder_model.get_output_shape_at(0)[0][1]

        encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length),
            dtype='float32')
        for i, input_text in enumerate(input_texts):
            print(input_text)
            input_text = self.start_token + " " + input_text+ " " + self.end_token

            for t, char in enumerate(input_text.split()):
                if char in self.input_token_index.keys():
                    encoder_input_data[i, t] = self.input_token_index[char]



        # Generate empty target sequence of length 1.
        target_seq = np.zeros(
                (1, self.max_decoder_seq_length),
                dtype='float32')
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0 ] = self.target_token_index[self.start_token]
        #decoder_input_data = [target_seq]

        states_value = self.encoder_model.predict(encoder_input_data)
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        predicted_count=0
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            #decoder_input_data = [target_seq]
            #output_tokens = model.predict([encoder_input_data,target_seq])
            # Sample a token
            predicted_count = predicted_count + 1
            pdf = output_tokens[0, predicted_count, :]
            sampled_token_index = np.random.choice(len(pdf),p=pdf)
            #sampled_token_index = np.argmax(pdf)
            sampled_char = self.reverse_target_char_index[str(sampled_token_index)]
            decoded_sentence += sampled_char + " "


            # Exit condition: either hit max length
            # or find stop character.
            if ('END' in sampled_char or sampled_char == '?' or sampled_char == '.' or
               len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True


            # Update the target sequence (of length 1).
            target_seq[0, predicted_count] = sampled_token_index


            # Update states
            states_value = [h, c]


        return decoded_sentence


