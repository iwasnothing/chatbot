from __future__ import print_function
import pandas as pd
import numpy as np
import os, sys
import json

def mkdirQA(dirkey):
    print(dirkey)
    try:
        if not os.path.exists(dirkey):
            os.makedirs(dirkey)
    except OSError as err:
        print(err)
        return

def printList(list,name):
    with open(name,"w") as f:
        for line in list:
            f.write(line + '\n')

start_token = "START "
end_token = " END"

with open('../train-v2.0.json') as f:
    data = json.load(f)

AList=[]
QList=[]
i=0
j=0
for d in data['data'][:-1]:
    for p in d['paragraphs']:
        for q in p['qas']:
            qn = start_token + q['question'].lower() + end_token
            QList.append(qn)
            ans=q['answers']
            if len(ans) > 0:
                a = ans[0]
                txt=start_token + a['text'].lower() + end_token
                AList.append(txt)
    i = i + 1
    if i%100 == 0 :
        dirkey = 'Q' +  str(i)
        mkdirQA(dirkey)
        printList(QList,dirkey+'/QList.txt')                
        printList(AList,dirkey+'/AList.txt')                
        QList = []
        AList = []
        j = i

if ( i > j):
    dirkey = 'Q' +  str(i)
    mkdirQA(dirkey)
    printList(QList,dirkey+'/QList.txt')                
    printList(AList,dirkey+'/AList.txt')                
    printList(QList,'QList')                
    printList(AList,'AList')                
    QList = []
    AList = []
