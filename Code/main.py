import numpy as np
# import matplotlib.pyplot as plt
import random
import os
import cv2
import tensorflow as tf
import time
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model
from keras.layers import Input, Concatenate, Dense
from keras.utils import plot_model
from manipulations import _extract_random_crop_edge, _extract_patches
from model import Gclass, Fpatch, Fglobal, Dissecting_image_crops_loss, ExpScheduler

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

resume = False #Set this to false if you do not want the pretrained model. 

if not resume:
    print("Creating Model")
    '''Defining the Individual Models'''
    Fglobal = Fglobal()
    Fpatch = Fpatch()
    Gmodel = Gclass()
    
    '''Stitching the models together'''
    input_layer = Input(shape=(224, 149, 3), name="Global_input")
    thumbnail_output = Fglobal(input_layer)
    
    '''Patch Embedding Model'''
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
    
    
    '''Patch Location Detection Model'''
    patch_net_input_01 = Input(shape=64, name='patch_net_input_01')
    patch_net_01 = Dense(16, activation='sigmoid', name="patch_net_dense_01")(patch_net_input_01)
    patch_net_input_02 = Input(shape=64, name='patch_net_input_02')
    patch_net_02 = Dense(16, activation='sigmoid', name="patch_net_dense_02")(patch_net_input_02)
    patch_net_input_03 = Input(shape=64, name='patch_net_input_03')
    patch_net_03 = Dense(16, activation='sigmoid', name="patch_net_dense_03")(patch_net_input_03)    
    patch_net_input_04 = Input(shape=64, name='patch_net_input_04')
    patch_net_04 = Dense(16, activation='sigmoid', name="patch_net_dense_04")(patch_net_input_04)    
    patch_net_input_05 = Input(shape=64, name='patch_net_input_05')
    patch_net_05 = Dense(16, activation='sigmoid', name="patch_net_dense_05")(patch_net_input_05)
    patch_net_input_06 = Input(shape=64, name='patch_net_input_06')
    patch_net_06 = Dense(16, activation='sigmoid', name="patch_net_dense_06")(patch_net_input_06)
    patch_net_input_07 = Input(shape=64, name='patch_net_input_07')
    patch_net_07 = Dense(16, activation='sigmoid', name="patch_net_dense_07")(patch_net_input_07)
    patch_net_input_08 = Input(shape=64, name='patch_net_input_08')
    patch_net_08 = Dense(16, activation='sigmoid', name="patch_net_dense_08")(patch_net_input_08) 
    patch_net_input_09 = Input(shape=64, name='patch_net_input_09')
    patch_net_09 = Dense(16, activation='sigmoid', name="patch_net_dense_09")(patch_net_input_09) 
    patch_net_input_10 = Input(shape=64, name='patch_net_input_10')
    patch_net_10 = Dense(16, activation='sigmoid', name="patch_net_dense_10")(patch_net_input_10)     
    patch_net_input_11 = Input(shape=64, name='patch_net_input_11')
    patch_net_11 = Dense(16, activation='sigmoid', name="patch_net_dense_11")(patch_net_input_11) 
    patch_net_input_12 = Input(shape=64, name='patch_net_input_12')
    patch_net_12 = Dense(16, activation='sigmoid', name="patch_net_dense_12")(patch_net_input_12) 
    patch_net_input_13 = Input(shape=64, name='patch_net_input_13')
    patch_net_13 = Dense(16, activation='sigmoid', name="patch_net_dense_13")(patch_net_input_13) 
    patch_net_input_14 = Input(shape=64, name='patch_net_input_14')
    patch_net_14 = Dense(16, activation='sigmoid', name="patch_net_dense_14")(patch_net_input_14) 
    patch_net_input_15 = Input(shape=64, name='patch_net_input_15')
    patch_net_15 = Dense(16, activation='sigmoid', name="patch_net_dense_15")(patch_net_input_15) 
    patch_net_input_16 = Input(shape=64, name='patch_net_input_16')
    patch_net_16 = Dense(16, activation='sigmoid', name="patch_net_dense_16")(patch_net_input_16) 
    
    
    '''Concatenated output of thumbnail + patches to feed into Crop Detection Model'''
    concatenated = Concatenate(name='Concatenated')([thumbnail_output, patch_output_01, patch_output_02, patch_output_03, patch_output_04,
                                                    patch_output_05, patch_output_06, patch_output_07, patch_output_08, 
                                                    patch_output_09, patch_output_10, patch_output_11, patch_output_12,
                                                    patch_output_13, patch_output_14, patch_output_15, patch_output_16])
    # output_layer = Gmodel(concatenated)
    
    # CropDetectionModel = Model(inputs=[input_layer, patch_input_01, patch_input_02, patch_input_03, patch_input_04,
    #                                                 patch_input_05, patch_input_06, patch_input_07, patch_input_08,
    #                                                 patch_input_09, patch_input_10, patch_input_11, patch_input_12,
    #                                                 patch_input_13, patch_input_14, patch_input_15, patch_input_16] , outputs=output_layer, name="CropDetectionModel")
    
    
    output_layer = Gmodel(thumbnail_output)
    CropDetectionModel = Model(inputs=input_layer, outputs=output_layer, name="CropDetectionModel")
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0000001)
    CropDetectionModel.compile(optimizer=opt, loss=Dissecting_image_crops_loss)
    # CropDetectionModel.compile(optimizer='adam', loss=Dissecting_image_crops_loss)
else:
    CropDetectionModel = load_model(r'C:\VISION\Dissecting Image Crops\Model', custom_objects={'Dissecting_image_crops_loss' : Dissecting_image_crops_loss})


CropDetectionModel.summary()

'''Making the Models visible'''
plot_model(Fglobal, to_file="C:\VISION\Dissecting Image Crops\Imgs\ResNet34_model.png", show_shapes=True)
plot_model(Fpatch, to_file="C:\VISION\Dissecting Image Crops\Imgs\ResNet18_model.png", show_shapes=True)
plot_model(Gmodel, to_file="C:\VISION\Dissecting Image Crops\Imgs\Perceptral_model.png", show_shapes=True)
plot_model(CropDetectionModel, to_file="C:\VISION\Dissecting Image Crops\Imgs\Merged_model.png", show_shapes=True)

train_dir = r'C:\VISION\Dissecting Image Crops\Data\train' 
test_dir  = r'C:\VISION\Dissecting Image Crops\Data\test'
val_dir   = r'C:\VISION\Dissecting Image Crops\Data\val'

filenames = os.listdir(train_dir)
BATCH_SIZE = 64

epochs = range(1)

for epoch in epochs:
    print("\n==================================================================")
    print("                  Starting Epoch: " + str(epoch + 1))
    print("==================================================================")
    for i in range(BATCH_SIZE, len(filenames), BATCH_SIZE):
        
        '''
        input_data is an array of dicts:
            dict = {
                'thumbnail'    = np.array([thumbnail])                            | Single downsized image (224, 149)
                'ground_truth' = [0, original_width, 0, original_height, cropped] | The original resolution with a bool stating if the img has been cropped
                'patches'      = [ np.array([patch]) , np.array([patch]), .... ]  | Array of all the patches
                'patch_loc'    = [0..16]                                          | Probability Distribution describing the estimated locations
            }
        '''
        
        input_data = []
        print()
        print("Loading Images")
        crops = 0
        t1 = time.time()
        for file in filenames[i-BATCH_SIZE:i]:
            data = dict()
            
            img = load_img(train_dir+'\\'+file)
            try:
                img = img_to_array(img)  
            except:
                print("Could not open file: "+str(file))
                continue
    
            chance = random.randint(0,100)
            if(chance < 80):
                crops += 1
                '''Crop the image'''
                crop, bounds, bounds_px1, size_factor = _extract_random_crop_edge(img, 0.5, 0.9, None, False)
                data['thumbnail'] = cv2.resize(crop, dsize=(149, 224))
                data['ground_truth'] = bounds
                data['ground_truth'].append(size_factor)

                img = crop
                
            else: 
                '''Keep the normal resolution'''
                data['thumbnail'] = cv2.resize(img, dsize=(149, 224))
                data['ground_truth'] = [0, 1, 0, 1, 0]
    
            # data['patches'], data['patch_loc'] = _extract_patches(img, 96, (224, 149))

            input_data.append(data)
            del img
        t2 = time.time()
        print("Loading images took: " + str(round((t2 - t1), 2)) + " seconds")
        print(str(crops) + " out of " + str(BATCH_SIZE) + " are cropped.")
        thumbnails    = np.asarray([d['thumbnail']    for d in input_data], dtype='object').astype(np.float32)
        ground_truths = np.asarray([d['ground_truth'] for d in input_data], dtype='object').astype(np.float32)

        print("Training model with batch number: " + str(int(i/BATCH_SIZE)) + " of " +str(int(len(filenames) / BATCH_SIZE)) + " batches." )
        CropDetectionModel.fit(thumbnails, ground_truths, epochs=1)
        
        ###Patch network additions
        # input_X   = []
        # patches   = np.empty([BATCH_SIZE, 16, 224, 149, 3])
        # patch_loc = np.empty([BATCH_SIZE, 16, 16])
        # for i in range(BATCH_SIZE):
        #     patches[i]   = (input_data[i]['patches'])
        #     patch_loc[i] = (input_data[i]['patch_loc'])
        #     line = []
        #     line.append(thumbnails[i])
        #     line.extend(patches[i])
        #     input_X.append([line])

        # x=[thumbnails, patches[:][0], patches[:][1], patches[:][2], patches[:][3],
        #         patches[:][4], patches[:][5], patches[:][6], patches[:][7],
        #         patches[:][8], patches[:][9], patches[:][10], patches[:][11],
        #         patches[:][12], patches[:][13], patches[:][14], patches[:][15]]
        
        # y=ground_truths
        # CropDetectionModel.fit(x=input_X, y=y,
                                # epochs=1)
    
CropDetectionModel.save(r'C:\VISION\Dissecting Image Crops\Model')
    
 
'''
#TODO:        
    Exponentially Decreasing Learning Rate for Adam Optimizer (min 5 * 10 ^-3 to max 1.5 * 10 ^-3)  (LearningRateScheduler) https://keras.io/api/callbacks/learning_rate_scheduler/
'''