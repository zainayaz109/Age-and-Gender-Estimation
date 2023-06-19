import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense,Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, GlobalAveragePooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate
from tensorflow.keras.regularizers import l2

root_dir = os.path.split(os.path.abspath(__file__))[0]

class MobileNetGender:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def __call__(self):
        """ Input """
        inputs = Input(shape=self.input_shape) # Defining Input Tensor

        """ Pre-trained MobileNet """                
        base_model = MobileNet(include_top=False, 
                              weights= root_dir + "/weights/mobilenet_1_0_224_tf_no_top.h5",
                              input_tensor=inputs)

        for layer in base_model.layers:
            layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512,activation='relu', kernel_regularizer=l2(0.005))(x)
        output = Dense(2, activation='softmax')(x)

        model = Model(inputs, output, name="MobileNet")
        
        return model