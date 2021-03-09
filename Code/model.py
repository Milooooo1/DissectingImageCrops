from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.models import Model
import tensorflow as tf
import numpy as np

def Dissecting_image_crops_loss(y_actual, y_predicted):
    '''
    This function calculates the loss for each part of the model
    and returns all of it. 
    
    LClass = Binary Cross Entropy: https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a , https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class
    LRect  = Mean Squared Error:   https://keras.io/api/losses/regression_losses/#mean_squared_error-function
    '''
    
    '''Crop Loss'''
    c_actual = y_actual[len(y_actual)-1]
    c_predicted = y_predicted[len(y_predicted)-1]
    bce       = tf.keras.losses.BinaryCrossentropy()
    LossClass = bce(c_actual, c_predicted)
    
    
    '''Thumbnail Loss'''
    mse         = tf.keras.losses.MeanSquaredError()
    LossRect    = mse(y_actual[:len(y_actual)-1], y_predicted[:len(y_predicted)-1])
    
    '''Patch Loss'''
    # LossPatch =

    totalLoss = (LossClass + LossRect)
    
    return totalLoss
    

'''
https://keras.io/guides/functional_api/#manipulate-complex-graph-topologies
https://keras.io/guides/functional_api/#all-models-are-callable-just-like-layers
'''


def Fpatch():
    '''
    https://arxiv.org/pdf/1512.03385.pdf>%60_
    ResNet-18 model that converts any patch into a length-64 embedding
    which then gets converted by a single linear layer on top to a 
    length-16 probability distribution describing the estimated
    location (^ik, ^jk) = {0...3}^2 of that patch.
    '''
    patch_size = (64,64,3)
        
    input_layer = Input(shape=patch_size, name="ResNet18_Input")
    x = Conv2D(64, (7, 7), strides=2, input_shape=patch_size, padding='same', name="ResNet18_conv0")(input_layer)
    x = MaxPooling2D((3,3), 2)(x)
    
    x = Conv2D(64, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv1")(x)
    x = Conv2D(64, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv2")(x)
    x = Conv2D(64, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv3")(x)
    x = Conv2D(64, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv4")(x)    
    
    x = Conv2D(128, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv5")(x)
    x = Conv2D(128, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv6")(x)
    x = Conv2D(128, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv7")(x)
    x = Conv2D(128, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv8")(x)

    x = Conv2D(256, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv9")(x)
    x = Conv2D(256, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv10")(x)
    x = Conv2D(256, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv11")(x)
    x = Conv2D(256, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv12")(x)
    
    x = Conv2D(512, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv13")(x)
    x = Conv2D(512, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv14")(x)
    x = Conv2D(512, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv15")(x)
    x = Conv2D(512, (3, 3), input_shape=patch_size, padding='same', name="ResNet18_conv16")(x)
    
    x = GlobalAveragePooling2D(name="ResNet18_GlobalAvgPooling")(x)
    
    x = Dense(1000, activation='softmax', name="ResNet18_Dense0")(x)
    
    x = Dense(64, activation='softmax', name="ResNet18_Dense1")(x)
    
    output_layer = Dense(16, activation='softmax', name="ResNet18_Output")(x) #16 outputs for location of patch
    
    fpatch = Model(inputs=input_layer, outputs=output_layer)
    
    return fpatch
    
def Fglobal():
    '''
    https://arxiv.org/pdf/1512.03385.pdf>%60_
    ResNet-34 model that converts the downscaled (224, 149) global image 
    into another length-64 embedding.
    '''
    downscaled_res = (224, 149, 3)   
    
    input_layer = Input(shape=downscaled_res, name="ResNet34_Input")
    x = Conv2D(64, (7, 7), strides=2, input_shape=downscaled_res, padding='same', name="ResNet34_conv0")(input_layer)
    x = MaxPooling2D((3,3), 2, name="ResNet34_MaxPool")(x) 
    
    x = Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv1")(x)
    x = Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv2")(x)
    x = Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv3")(x)
    x = Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv4")(x)
    x = Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv5")(x)
    x = Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv6")(x)
    
    x = Conv2D(128, (3, 3), strides=2, input_shape=downscaled_res, padding='same', name="ResNet34_conv7")(x)
    x = Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv8")(x)
    x = Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv9")(x)
    x = Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv10")(x)
    x = Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv11")(x)
    x = Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv12")(x)
    x = Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv13")(x)
    x = Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv14")(x)
    
    x = Conv2D(256, (3, 3), strides=2, input_shape=downscaled_res, padding='same', name="ResNet34_conv16")(x)
    x = Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv17")(x)
    x = Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv18")(x)
    x = Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv19")(x)
    x = Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv20")(x)
    x = Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv21")(x)
    x = Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv22")(x)
    x = Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv23")(x)
    x = Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv24")(x)
    x = Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv25")(x)
    x = Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv26")(x)
    
    x = Conv2D(512, (3, 3), strides=2, input_shape=downscaled_res, padding='same', name="ResNet34_conv27")(x)
    x = Conv2D(512, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv28")(x)
    x = Conv2D(512, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv29")(x)
    x = Conv2D(512, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv30")(x)
    x = Conv2D(512, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv31")(x)
    x = Conv2D(512, (3, 3), input_shape=downscaled_res, padding='same', name="ResNet34_conv32")(x)
    
    x = GlobalAveragePooling2D(name="ResNet34_GlobalAvgPooling")(x)
    
    x = Dense(1000, activation='softmax', name="ResNet34_Dense")(x)
    
    output_layer = Dense(64, activation='softmax', name="ResNet34_Output")(x)
    
    fglobal = Model(inputs=input_layer, outputs=output_layer, name="ResNet34_Model")
    
    return fglobal

    # return output_layer

def Gclass():
    '''
    3-layer Perceptron that accepts a 1088-dimensional concatenation 
    of all previous embeddings (input_layer) and produces 5 values 
    describing the crop rectangle (^x1, ^x2, ^y1, ^y2) = [0,1]^4 and 
    the actual probability ^c that the input image had been cropped. 
    '''
    
    input_layer = Input(shape=64, name="Perceptral_Input")
    x = Dense(512, activation='sigmoid', name="Perceptral_Dense1")(input_layer)
    x = Dense(265, activation='sigmoid', name="Perceptral_Dense2")(x)
    output_layer = Dense(5, activation='sigmoid', name="Perceptral_Output")(x)

    gclass = Model(inputs=input_layer, outputs=output_layer, name="Perceptral_Model")    
    return gclass
    



































