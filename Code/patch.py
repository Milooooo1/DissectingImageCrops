# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:25:56 2021

@author: Milo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

import random
import os
import tensorflow as tf
import time

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import Model, load_model
from keras.layers import Input
from keras.utils import plot_model

from manipulations import _extract_patches, _extract_random_crop_edge
from model import Fpatch 

#Code van de tensorflow site om op de GPU te trainen
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print("ERROR: " + str(e))
        


train_dir = r'C:\VISION\Dissecting Image Crops\Data\train' 
test_dir  = r'C:\VISION\Dissecting Image Crops\Data\test'
val_dir   = r'C:\VISION\Dissecting Image Crops\Data\val'

train = True        
resume = False

if train:
    if not resume:
        
        Fpatch = Fpatch()
        
        '''Patch Network inputs'''
        patch_input_01 = Input(shape=(96, 96, 3), name="Patch_01_input")
        patch_output_01 = Fpatch(patch_input_01)
        patch_input_02 = Input(shape=(96, 96, 3), name="Patch_02_input")
        patch_output_02 = Fpatch(patch_input_02)
        patch_input_03 = Input(shape=(96, 96, 3), name="Patch_03_input")
        patch_output_03 = Fpatch(patch_input_03)
        patch_input_04 = Input(shape=(96, 96, 3), name="Patch_04_input")
        patch_output_04 = Fpatch(patch_input_04)
        patch_input_05 = Input(shape=(96, 96, 3), name="Patch_05_input")
        patch_output_05 = Fpatch(patch_input_05)
        patch_input_06 = Input(shape=(96, 96, 3), name="Patch_06_input")
        patch_output_06 = Fpatch(patch_input_06)
        patch_input_07 = Input(shape=(96, 96, 3), name="Patch_07_input")
        patch_output_07 = Fpatch(patch_input_07)
        patch_input_08 = Input(shape=(96, 96, 3), name="Patch_08_input")
        patch_output_08 = Fpatch(patch_input_08)
        patch_input_09 = Input(shape=(96, 96, 3), name="Patch_09_input")
        patch_output_09 = Fpatch(patch_input_09)
        patch_input_10 = Input(shape=(96, 96, 3), name="Patch_10_input")
        patch_output_10 = Fpatch(patch_input_10)
        patch_input_11 = Input(shape=(96, 96, 3), name="Patch_11_input")
        patch_output_11 = Fpatch(patch_input_11)
        patch_input_12 = Input(shape=(96, 96, 3), name="Patch_12_input")
        patch_output_12 = Fpatch(patch_input_12)
        patch_input_13 = Input(shape=(96, 96, 3), name="Patch_13_input")
        patch_output_13 = Fpatch(patch_input_13)
        patch_input_14 = Input(shape=(96, 96, 3), name="Patch_14_input")
        patch_output_14 = Fpatch(patch_input_14)
        patch_input_15 = Input(shape=(96, 96, 3), name="Patch_15_input")
        patch_output_15 = Fpatch(patch_input_15)
        patch_input_16 = Input(shape=(96, 96, 3), name="Patch_16_input")
        patch_output_16 = Fpatch(patch_input_16)
        
        PatchNetModel = Model(inputs=[ patch_input_01, patch_input_02, patch_input_03, patch_input_04,
                                       patch_input_05, patch_input_06, patch_input_07, patch_input_08,
                                       patch_input_09, patch_input_10, patch_input_11, patch_input_12,
                                       patch_input_13, patch_input_14, patch_input_15, patch_input_16
                                     ],
                              outputs=[patch_output_01, patch_output_02, patch_output_03, patch_output_04,
                                       patch_output_05, patch_output_06, patch_output_07, patch_output_08,
                                       patch_output_09, patch_output_10, patch_output_11, patch_output_12,
                                       patch_output_13, patch_output_14, patch_output_15, patch_output_16
                                      ], 
                              name="PatchNetwork")
        
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        PatchNetModel.compile(loss="categorical_crossentropy", optimizer=opt)#, metrics=['accuracy'])    
    else:
        PatchNetModel = load_model(r'C:\VISION\Dissecting Image Crops\Checkpoints\Patch\Model')#, custom_objects={'Dissecting_image_crops_loss' : Dissecting_image_crops_loss})    
          
    filenames = os.listdir(train_dir)
    BATCH_SIZE = 16
    
    epochs = range(1)
    
    for epoch in epochs:
        print("\n==================================================================")
        print("                  Starting Epoch: " + str(epoch + 1))
        print("==================================================================")
        for i in range(BATCH_SIZE, len(filenames), BATCH_SIZE):
           
            input_data = []
            label_data = []
            print()
            print("Loading Images")
            crops = 0
            t1 = time.time()
            for file in filenames[i-BATCH_SIZE:i]:
                
                img = load_img(train_dir+'\\'+file)
                try:
                    img = img_to_array(img)  
                except:
                    print("Could not open file: "+str(file))
                    continue
        
                chance = random.randint(0,100)
                if(chance < 50):
                    crops += 1
                    '''Crop the image'''
                    crop, bounds, bounds_px1, size_factor = _extract_random_crop_edge(img, 0.5, 0.9, None, False)
                    img = crop                    
    
                patch_arr, positions = _extract_patches(img)
                
                input_data.append(patch_arr)
                label_data.append(positions)
                
                
            t2 = time.time()
            print("Loading images took: " + str(round((t2 - t1), 2)) + " seconds")
            print(str(crops) + " out of " + str(BATCH_SIZE) + " are cropped.")
            print("Training model with batch number: " + str(int(i/BATCH_SIZE)) + " of " +str(int(len(filenames) / BATCH_SIZE)) + " batches." )
            PatchNetModel.fit(x=input_data[:16], y=label_data[:16])
            
            
    PatchNetModel.save(r'C:\VISION\Dissecting Image Crops\Checkpoints\Patch')    


else:
    '''Test the model'''
    PatchNetModel = load_model(r'C:\VISION\Dissecting Image Crops\Checkpoints\Patch\\')       
    print("Model Loaded") 
    
    filenames = os.listdir(test_dir)
    BATCH_SIZE = 16
    
    for i in range(BATCH_SIZE, len(filenames), BATCH_SIZE):

        input_data = []
        label_data = []
        print()
        print("Loading Images")

        for file in filenames[i-BATCH_SIZE:i]:
            
            img = load_img(test_dir+'\\'+file)
            try:
                img = img_to_array(img)  
            except:
                print("Could not open file: "+str(file))
                continue


            patch_arr, positions = _extract_patches(img)
            
            input_data.append(patch_arr)
            label_data.append(positions)
    
        
        result = PatchNetModel.predict(x=input_data[:16],
                                        batch_size=None)
        print(result)
        break

    pos_list = [(-0.5, -0.5), (0.5, -0.5), (1.5, -0.5), (2.5, -0.5),
                (-0.5,  0.5), (0.5,  0.5), (1.5,  0.5), (2.5,  0.5),
                (-0.5,  1.5), (0.5,  1.5), (1.5,  1.5), (2.5,  1.5),
                (-0.5,  2.5), (0.5,  2.5), (1.5,  2.5), (2.5,  2.5),
                ]
    for i in range(0, BATCH_SIZE):              
        res = np.reshape(result[0][i], (4,4))       
        
        nodes = [0, 0.2, 0.4, 1.0]
        colors = ["purple", "blue", "green", "yellow"]
        cmap = LinearSegmentedColormap.from_list("", list(zip(nodes, colors)))
        cmap.set_under("gray")
            
        fig, ax2 = plt.subplots()    
        ax2.axis('off')
        ax2.add_patch(patches.Rectangle(pos_list[i], 1, 1, edgecolor='red', fill=False, linewidth=2))
        im = ax2.imshow(res, cmap=cmap)
        fig.colorbar(im, extend="min") 
        plt.show()
    
        fig, axs = plt.subplots()   
        axs.axis('off')
        im = axs.imshow(array_to_img(input_data[0][i]))
        plt.show()









































    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    