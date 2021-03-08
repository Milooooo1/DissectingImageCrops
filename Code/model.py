from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential

def Dissecting_image_crops_loss(y_actual, y_predicted):
    pass


def Fpatch():
    '''
    https://arxiv.org/pdf/1512.03385.pdf>%60_
    https://keras.io/api/layers/merging_layers/concatenate/
    ResNet-18 model that converts any patch into a length-64 embedding
    which then gets converted by a single linear layer on top to a 
    length-16 probability distribution describing the estimated
    location (^ik, ^jk) = {0...3}^2 of that patch.
    '''
    patch_size = (64,64)
    
    model = Sequential([])
    
    model.add(Conv2D(64, (7, 7), strides=2, input_shape=patch_size, padding='same'))
    model.add(MaxPooling2D((3,3), 2))
    
    model.add(Conv2D(64, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=patch_size, padding='same'))    
    
    model.add(Conv2D(128, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(128, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(128, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(128, (3, 3), input_shape=patch_size, padding='same'))

    model.add(Conv2D(256, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=patch_size, padding='same'))
    
    model.add(Conv2D(512, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(512, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(512, (3, 3), input_shape=patch_size, padding='same'))
    model.add(Conv2D(512, (3, 3), input_shape=patch_size, padding='same'))
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(1000, activation='softmax'))
    
    model.add(Dense(64, actication='softmax'))
    
    model.add(Dense(16, activation='softmax')) #16 outputs for location of patch
    
    return model
    
def Fglobal():
    '''
    https://arxiv.org/pdf/1512.03385.pdf>%60_
    ResNet-34 model that converts the downscaled (224, 149) global image 
    into another length-64 embedding.
    '''
    downscaled_res = (224, 149, 3)
    
    model = Sequential([])
    
    model.add(Conv2D(64, (7, 7), strides=2, input_shape=downscaled_res, padding='same'))
    model.add(MaxPooling2D((3,3), 2)) 
    
    model.add(Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=downscaled_res, padding='same'))
    
    model.add(Conv2D(128, (3, 3), strides=2, input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(128, (3, 3), input_shape=downscaled_res, padding='same'))
    
    model.add(Conv2D(256, (3, 3), strides=2, input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(256, (3, 3), input_shape=downscaled_res, padding='same'))
    
    model.add(Conv2D(512, (3, 3), strides=2, input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(512, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(512, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(512, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(512, (3, 3), input_shape=downscaled_res, padding='same'))
    model.add(Conv2D(512, (3, 3), input_shape=downscaled_res, padding='same'))
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(1000, activation='softmax'))
    
    model.add(Dense(64, activation='softmax'))
    
    return model

def Gclass():
    '''
    3-layer Perceptron that accepts a 1088-dimensional concatenation 
    of all previous embeddings and produces 5 values describing the 
    crop rectangle (^x1, ^x2, ^y1, ^y2) = [0,1]^4 and the actual
    probability ^c that the input image had been cropped. 
    '''
    model = Sequential([])
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(5, activation='sigmoid'))
    
    return model
    



































