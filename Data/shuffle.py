# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 22:22:02 2021

@author: Milo
"""

import os
import random

'''MAKE SURE TO CHANGE THESE DIRECTORIES'''
train_dir = r'C:\VISION\dissecting-image-crops\data\train'
test_dir  = r'C:\VISION\dissecting-image-crops\data\test'
val_dir   = r'C:\VISION\dissecting-image-crops\data\val'

data_dir  = r'C:\VISION\dissecting-image-crops\data'

dataset = []

for file in os.listdir(data_dir):
    name = os.fsdecode(file)
    if( name.endswith(".jpg") ):
        dataset.append(name)
    
random.shuffle(dataset)

print(len(dataset))

trainPercentage = int(len(dataset) * 0.75)
testPercentage  = int(len(dataset) * 0.15)
valPercentage   = int(len(dataset) * 0.10)

print(trainPercentage)
print(testPercentage)
print(valPercentage)

for i in range(0, trainPercentage):
    os.rename(data_dir+"\\"+dataset[i], train_dir+"\\"+dataset[i])

for i in range(trainPercentage, trainPercentage+testPercentage):
    os.rename(data_dir+"\\"+dataset[i], test_dir+"\\"+dataset[i])
    
for i in range(trainPercentage+testPercentage, trainPercentage+testPercentage+valPercentage):
    os.rename(data_dir+"\\"+dataset[i], val_dir+"\\"+dataset[i])