import math
from sklearn.model_selection import train_test_split
import numpy as np
from keras import backend as K
from keras.layers import *
from keras.models import *
import math
import tensorflow as tf

#device = ("cuda" if torch.cuda.is_available() else "cpu")

def func(x1, x2):
  A = -(x2+47)*math.sin(math.sqrt(math.fabs((x1/2) + (x2+47))))
  B = -x1*math.sin(math.sqrt(math.fabs(x1-(x2+47))))
  return A + B

def make_dataset():
  samples = np.random.normal(0, 0.3, 100000)
  xs = (np.random.rand(100000,2) - 0.5)*2*512
  f = []

  for i in range(len(xs)):
    temp = func(xs[i][0], xs[i][1])
    f.append(temp)

  f = np.array(f)
  y = f + samples

  x_train, x_test, y_train, y_test = train_test_split(xs, y, test_size=0.2)

  x_train = x_train.reshape(80000, 2, 1)
  x_test = x_test.reshape(20000, 2, 1)

  return x_train, x_test, y_train, y_test

def rmse(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))



file = open("3layers.txt", 'w')
x_train, x_test, y_train, y_test = make_dataset()
for ele in [16, 32, 64, 128]:
    model = Sequential()
    model.add(Dense(ele, input_dim = 2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(2*ele, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(ele, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', rmse])
    save_history = model.fit(x_train, y_train, epochs=1750, batch_size=32, validation_split=0.2)
    metrics = model.evaluate(x_test, y_test)
    print("Metrics for ", ele, " units hidden layer are: ", metrics)
    file.write(metrics)
    file.write("\n")

file.close()
