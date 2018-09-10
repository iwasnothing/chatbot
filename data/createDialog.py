from __future__ import print_function
import pandas as pd
import os, sys
import numpy as np

import json

start_token = "START"
end_token = "END"

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
            f.write((line + '\n').encode("utf8").decode("cp950", "ignore"))

AList=[]
QList=[]
source = str(sys.argv[1])
with open(source, encoding = 'utf8') as f:
    lines = f.read()[:-1].split('\n')
    
    #lines = [start_token+name.lower()+end_token for name in lines]
    
print ('n samples = ',len(lines))
j=0
i=0
for a in range(len(lines)):
    line2 = lines[a].split('|')
    input_text  = start_token+" "+line2[0]+" "+end_token
    if type(input_text) is not str:
        continue
    QList.append(input_text)
        
    target_text =  start_token+" "+line2[1]+" "+end_token
    if type(target_text) is not str:
        continue
    AList.append(target_text)
    i = i + 1
    if i % 10000 == 0 :
        dirkey = 'T' +  str(i)
        mkdirQA(dirkey)
        printList(QList,dirkey+'/QList.txt')                
        printList(AList,dirkey+'/AList.txt')                
        QList = []
        AList = []
        b = a

if ( i > j):
    dirkey = 'P' +  str(i)
    mkdirQA(dirkey)
    printList(QList,dirkey+'/QList.txt')                
    printList(AList,dirkey+'/AList.txt')                
    printList(QList,'QList')                
    printList(AList,'AList')                
    QList = []
    AList = []

    
