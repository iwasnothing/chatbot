from __future__ import print_function
import pandas as pd

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

import json

start_token = "START "
end_token = " END"

with open('/Users/kahingleung/Downloads/train-v2.0.json') as f:
    data = json.load(f)

AList=[]
QList=[]
for k, v in data['data'][:3].items():
    print(k, v)

for d in data['data'][:3]:
    for p in d['paragraphs']:
        for q in p['qas']:
            qn = start_token + q['question'].lower() + end_token
            QList.append(qn)
            ans=q['answers']
            if len(ans) > 0:
                a = ans[0]
                txt=start_token + a['text'].lower() + end_token
                AList.append(txt)
                
print(len(AList))
print(len(QList))


min_samples = min([len(QList), len(AList)])
num_samples = 500
rclist = np.random.choice(min_samples,num_samples,replace=False)
#for i in range(num_samples):
for i in rclist:
    input_text = QList[i]
    target_text = AList[i] 
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text.split():
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text.split():
        if char not in target_characters:
            target_characters.add(char)


#input_characters = sorted(list(input_characters))
#target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])


print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
with open('count.csv', 'a') as the_file:
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


def writeDic(data,name):
    with open(name + '.json', 'w') as outfile:
        json.dump(data, outfile)

writeDic(input_token_index,'input_token_index')
writeDic(target_token_index,'target_token_index')
writeDic(reverse_input_char_index,'reverse_input_char_index')
writeDic(reverse_target_char_index,'reverse_target_char_index')


