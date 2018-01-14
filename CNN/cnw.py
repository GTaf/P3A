#!/usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import os
#from osgeo import gdal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# In[2]:


batch_size = 5
num_classes = 2
epochs = 50

img_rows, img_cols = 81, 81

array = []
et = []

"""
# In[3]:
for element in os.listdir('Base_Defi/Fond/'):
    dataset = gdal.Open('Base_Defi/Fond/'+element, gdal.GA_ReadOnly)
    channel = np.array(dataset.GetRasterBand(1).ReadAsArray())
    if channel.shape == (81,81):
        array.append([channel.tolist()])
        array.append(np.flip([channel.tolist()],1))
        array.append(np.flip([channel.tolist()],0))
        array.append(np.flip(np.flip([channel.tolist()],0),1))
        et.append(1)
        et.append(1)
        et.append(1)
        et.append(1)
        
l = len(array)
print("taille de l'entrée fond:",l)
i = 0
for element in os.listdir('Base_Defi/Automobile/Valide/'):
     
     dataset = gdal.Open('Base_Defi/Automobile/Valide/'+element, gdal.GA_ReadOnly)
     channel = np.array(dataset.GetRasterBand(1).ReadAsArray())
     if channel.shape == (81,81):
        array.append([channel.tolist()])
        et.append(0)
        i = i+1
     if i == l:
         break
print("taille de l'entrée voiture:",len(array)-l)
print("elts valides", len(array))

"""
     


# In[6]:


x_train = np.load("x.npy")
y_train = np.load("y.npy")

#x_train = np.array(array)
#y_train = np.array(et)
#y_train = label_binarize(y_train, classes=[0, 1])
x_test = np.array([])
y_test = np.array([])
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.05)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)



print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(y_train)

from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

tot = 0
ref = y_train[0]
for i in range(len(y_train)):
    if (y_train[i] == ref).all():
        tot = tot + 1
        
print("proportion : ",tot/len(y_train),"      len(y_train)  ",len(y_train))


# In[7]:


def baseline_model():
    model = Sequential()

    model.add(Conv2D(32, (2, 2), input_shape=(81,81,1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (2, 2)))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
 
    BatchNormalization(axis=-1)
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
# 

    model.add(Flatten())
#    # Fully connected layer
#
    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.5))
    model.add(Dense(2))
    
    sgd = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',#keras.optimizers.Adam(),
              metrics=['accuracy'])
    
    return model

    # model.add(Convolution2D(10,3,3, border_mode='same'))
    # model.add(GlobalAveragePooling2D())
    #model.add(Activation('softmax'))

model = baseline_model()

# In[8]:


import pandas
dtype = [('Col1','int32'), ('Col2','float32'), ('Col3','float32')]
values = y_train
index = ['Row'+str(i) for i in range(1, len(values)+1)]

df = pandas.DataFrame(values, index=index)


y_classes = df.idxmax(1, skipna=False)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(list(y_classes))
y_integers = le.transform(list(y_classes))
sample_weights = class_weight.compute_sample_weight('balanced', y_integers)


# In[9]:

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# check 5 epochs
early_stop = EarlyStopping(monitor='val_acc', patience=20, mode='max') 

callbacks_list = [checkpoint, early_stop]



history = model.fit(x_train, y_train,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=callbacks_list
          #sample_weight = sample_weights
          )
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, x_train, dummy_y, cv=kfold,fit_params={'sample_weight': sample_weights})
"""

# In[14]:


print(history.history.keys())
#  "Accuracy"
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
# "Loss"
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()


# In[15]:


y_pred = model.predict(x_test)


# In[16]:


yy_test = []
yy_pred = []
for i in range(len(y_test)):
    if y_pred[i][0] == 0:
        yy_pred.append(1)
    else:
        yy_pred.append(0)
    
    if y_test[i][0] == 0:
        yy_test.append(1)
    else:
        yy_test.append(0)


# In[17]:


confusion_matrix(yy_test, yy_pred)



