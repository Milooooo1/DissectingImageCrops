import numpy as np
# import matplotlib.pyplot as plt
import random
import os
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from manipulations import _extract_random_crop_edge
from model import Gclass, Fpatch, Fglobal


Fglobal = Fglobal()

Fpatch_0  = Fpatch()
Fpatch_1  = Fpatch()
Fpatch_2  = Fpatch()
Fpatch_3  = Fpatch()
Fpatch_4  = Fpatch()
Fpatch_5  = Fpatch()
Fpatch_6  = Fpatch()
Fpatch_7  = Fpatch()
Fpatch_8  = Fpatch()
Fpatch_9  = Fpatch()
Fpatch_10 = Fpatch()
Fpatch_11 = Fpatch()
Fpatch_12 = Fpatch()
Fpatch_13 = Fpatch()
Fpatch_14 = Fpatch()
Fpatch_15 = Fpatch()

Gmodel = Gclass(inputs=Fglobal.outputs) #Concatenate all outputs https://stackoverflow.com/questions/45979848/merge-2-sequential-models-in-keras

train_dir = r'F:\flickr-scrape\images\sorted\train' 
test_dir  = r'F:\flickr-scrape\images\sorted\test'
val_dir   = r'F:\flickr-scrape\images\sorted\val'

filenames = os.listdir(train_dir)
BATCH_SIZE = 64

for i in range(BATCH_SIZE, len(filenames), BATCH_SIZE):
    '''
    input_data is an array of dicts:
        dict = {
            'thumbnail'    = np.array([thumbnail])                            | Single downsized image (224, 149)
            'ground_truth' = [0, 0, original_width, original_hight]           | Ground truth = the original resolution
            'patches'      = [ np.array([patch]) , np.array([patch]), .... ]  | Array of all the patches
            'patch_loc'    = [0..3]^2                                         | Probability Distribution describing the estimated locations
        }
    '''
    print("Training model with batch number: " + str(i/BATCH_SIZE) )
    
    input_data = []
    
    for file in filenames[i-BATCH_SIZE:i]:
        img = load_img(train_dir+'\\'+file)
        data = dict()
        data['ground_truth'] = img.shape
        
        if(random.randint(0,1) == 1):
            '''Crop the image and resize it with a width between 1024 2048'''
            img = img_to_array(img)
            crop, bounds, bounds_px1, size_factor = _extract_random_crop_edge(img, 0.5, 0.9, None, False)
            
            crop = array_to_img(crop)
            width = random.randint(1024,2048)
            crop.resize((width, int(width/1.5)))
            img = img_to_array(crop)
            
            data['thumbnail'] = img_to_array(crop.resize((224, 149)))
        else: 
            '''Keep the normal resolution'''
            data['thumbnail'] = img_to_array(img.resize((224, 149)))
        
        #TODO : Divide this crop (or not) in 16 patches 8+- jitter to it
    
    
    #TODO: Train Fglobal and Fpatch and pipe outputs into Gmodel
    
    input_data = np.array(input_data)
    print(type(input_data))
    print(len(input_data))
    
    
    
    
    
    
    
    
    
    
    
    
    
'''
#TODO:     
    Loss functie voor Fpatch en Frect
    
    Kijk hoe je patches met shared weight kan laten trainen. 
    
    Zoek optimalisatie uit met GPU
    
    Optimizations applied to the random crop:
        Positive (Outward) Red Transverse Chromatic Aberration
        Positive (Outward) Blue Transverse Chromatic Aberration
        
    
'''