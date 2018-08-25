from __future__ import print_function
import pandas as pd
import os, sys
import numpy as np

import json

start_token = "START "
end_token = " END"

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

AList=[]
QList=[]
with open("/Users/kahingleung/Downloads/talk6810.txt") as f:
    lines = f.read()[:-1].split('\n')
    lines = [start_token+name.lower()+end_token for name in lines]
    
print ('n samples = ',len(lines))
j=0
i=0
for a in range(len(lines)-1):
    input_text  = lines[a] 
    if type(input_text) is not str:
        continue
    QList.append(input_text)
        
    target_text = lines[a+1]
    if type(target_text) is not str:
        continue
    AList.append(target_text)
    i = i + 1
    if i % 100 == 0 :
        dirkey = 'T' +  str(i)
        mkdirQA(dirkey)
        printList(QList,dirkey+'/QList.txt')                
        printList(AList,dirkey+'/AList.txt')                
        QList = []
        AList = []
        b = a

if ( i > j):
    dirkey = 'T' +  str(i)
    mkdirQA(dirkey)
    printList(QList,dirkey+'/QList.txt')                
    printList(AList,dirkey+'/AList.txt')                
    printList(QList,'QList')                
    printList(AList,'AList')                
    QList = []
    AList = []

    
