import numpy as np
# import matplotlib.pyplot as plt
import random
import os
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model
from manipulations import _extract_random_crop_edge
from model import Gclass, Fpatch, Fglobal

'''Defining the Individual Models'''
Fglobal = Fglobal()
Fpatch = Fpatch()
Gmodel = Gclass()

'''Stitching the models together'''
input_layer = Input(shape=(224, 149, 3), name="global_input")
x = Fglobal(input_layer)
# x = Fpatch(x)
output_layer = Gmodel(x)

EindBeest = Model(inputs=input_layer, outputs=output_layer, name="EindBeest")
EindBeest.summary()

EindBeest.compile(optimizer="adam")

'''Making the Models visible'''
plot_model(Fglobal, to_file="C:\VISION\Dissecting Image Crops\Code\ResNet34_model.png", show_shapes=True)
plot_model(Gmodel, to_file="C:\VISION\Dissecting Image Crops\Code\Perceptral_model.png", show_shapes=True)
plot_model(EindBeest, to_file="C:\VISION\Dissecting Image Crops\Code\Merged_model.png", show_shapes=True)

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
    
    
    input_data = []
    print("Loading Images")
    for file in filenames[i-BATCH_SIZE:i]:
        data = dict()
        
        img = load_img(train_dir+'\\'+file)
        
        if(random.randint(0,1) == 1):
            '''Crop the image'''
            img = img_to_array(img)    
            crop, bounds, bounds_px1, size_factor = _extract_random_crop_edge(img, 0.5, 0.9, None, False)
            data['thumbnail'] = crop
            img = crop
            
        else: 
            '''Keep the normal resolution'''
            img = img_to_array(img.resize((224, 149)))
            data['thumbnail'] = img

        data['ground_truth'] = img.shape
        
        #TODO : Divide the remaining image in 16 patches with a 8+- jitter to it
    
        input_data.append(data)
    
    print("Training model with batch number: " + str(i/BATCH_SIZE) + " of " +str(int(len(filenames) / 64)) + " batches." )
    #TODO: Train Fglobal and Fpatch and pipe outputs into Gmodel
    thumbnails    = np.array([d['thumbnail']    for d in input_data])
    ground_truths = np.array([d['ground_truth'] for d in input_data])

    EindBeest.fit(thumbnails, ground_truths)
    # print(type(input_data))
    # print(len(input_data))
    # print(input_data[0])
    
    
    
    
    
    
    
    
    
    
    
    
    
'''
#TODO:     
    Loss functie voor Fpatch en Frect
    
    Kijk hoe je patches met shared weight kan laten trainen. 
    
    Zoek optimalisatie uit met GPU
    
    Optimizations applied to the random crop:
        Positive (Outward) Red Transverse Chromatic Aberration
        Positive (Outward) Blue Transverse Chromatic Aberration
        
    
'''