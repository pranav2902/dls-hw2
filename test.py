from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K
from keras.metrics import categorical_accuracy,top_k_categorical_accuracy

def conv_module(x,K,kX,kY, stride, chanDim, padding="same"):
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        return x

def inception_module(x, numK1x1, numK3x3, chanDim):
    conv_1x1 = conv_module(x, numK1x1, 1, 1,(1, 1), chanDim)
    conv_3x3 = conv_module(x, numK3x3, 3, 3,(1, 1), chanDim)
    x = concatenate([conv_1x1, conv_3x3], axis=chanDim)
    return x

def downsample_module(x, K, chanDim):
    conv_3x3 = conv_module(x, K, 3, 3, (2, 2),-1, padding="valid")
    pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate([conv_3x3, pool], axis=chanDim)
    return x

def build(width, height, depth, classes):
    inputShape = (height, width, depth)
    chanDim = -1
    
    inputs = Input(shape=inputShape)
    x = conv_module(inputs, 96, 3, 3, (1, 1),chanDim)
    x = inception_module(x, 32, 32, chanDim)
    x = inception_module(x, 32, 48, chanDim)
    x = downsample_module(x, 80, chanDim)
    x = inception_module(x, 112, 48, chanDim)
    x = inception_module(x, 96, 64, chanDim)
    x = inception_module(x, 80, 80, chanDim)
    x = inception_module(x, 48, 96, chanDim)
    x = downsample_module(x, 96, chanDim)
    x = inception_module(x, 176, 160, chanDim)
    x = inception_module(x, 176, 160, chanDim)
    x = AveragePooling2D((7, 7),padding = "same")(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(classes)(x)
    x = Activation("softmax")(x)
    model = Model(inputs, x, name="googlenet")
    return model

 def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)
  
optimizer = Adam(lr=0.01)
model = build(28,28,1,10)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc',top_5_accuracy])
hists = []
for i in [32,64,128,256,512,1024,2048,4096,8192,16384]:
    lr = i
    hist = model.fit(input_train, y_train, epochs = 5)
    hists.append(hist)
print(hists)
