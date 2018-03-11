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
from skimage.feature import canny

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# In[2]:


batch_size = 16
num_classes = 2
epochs = 40 

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
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

def gradient(image):
    cha1 = np.gradient(image,axis=(0,1))[1]
    cha2 = np.gradient(image,axis=(0,1))[0]
    return np.sqrt(np.power(cha1,2)+np.power(cha2,2))

#for i in range(len(x_train)):
#    x_train[i][0] = canny(x_train[i][0])



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


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
# In[7]:


def f1_score(y_true, y_pred):
    """
    f1 score

    :param y_true:
    :param y_pred:
    :return:
    """
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * ((precision * recall) / (precision + recall))


def simple_image_clf() :

    model = Sequential()
    
            # Convolution layers

    model.add(Conv2D(filters = 32, kernel_size = (5, 5), input_shape = (81, 81, 1), strides = (1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Dropout(0.5))
    
    model.add(Conv2D(filters = 64, kernel_size = (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Dropout(0.5))
    model.add(Conv2D(filters = 128, kernel_size = (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

                                                                # The binary classifier : 2 dense layers + a sigmoid top

    model.add(Flatten())

    model.add(Dense(units = 16))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(units = 16))
    model.add(Activation('relu'))
    model.add(Dense(units = 2))
    model.add(Activation('softmax'))


    from keras.optimizers import Adadelta
    from keras.metrics import binary_crossentropy, binary_accuracy

    model.compile(loss = binary_crossentropy,
                          optimizer = 'Nadam',#,Adadelta(lr = 1.0, rho = 0.95),
                        metrics=['accuracy'])

    return model




def baseline_model():
    model = Sequential()

    model.add(Conv2D(32, (2, 2), input_shape=(81,81,1)))
#    model.add(Activation('relu'))
#    BatchNormalization(axis=-1)
    model.add(MaxPooling2D(pool_size=(2,2)))
#    model.add(Conv2D(32, (2, 2)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (2, 2)))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
 
    BatchNormalization(axis=-1)
    model.add(Conv2D(64,(3, 3)))
#    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
#    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
# 

    model.add(Flatten())
#    # Fully connected layer
#
    BatchNormalization()
    model.add(Dense(512, activation = 'relu'))
    BatchNormalization()
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))
    
    sgd = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',#keras.optimizers.Adam(),
              metrics=['recall'])
    
    return model

    # model.add(Convolution2D(10,3,3, border_mode='same'))
    # model.add(GlobalAveragePooling2D())
    #model.add(Activation('softmax'))

model = simple_image_clf()

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
import datetime
now = datetime.datetime.now()
directory = now.strftime("%Y%m%d%H%M")
if not os.path.exists(directory):
        os.makedirs(directory)
filepath=directory + "/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=2, save_best_only=True, mode='max')

# check 5 epochs
early_stop = EarlyStopping(monitor='val_acc', patience=200, mode='max') 

callbacks_list = [checkpoint]

model_json = model.to_json()
with open(directory+"/model.json", "w") as json_file:
    json_file.write(model_json)

tb = keras.callbacks.TensorBoard(log_dir='./'+directory+'/Graph', histogram_freq=0,write_graph=True,write_images=True)


import sklearn.metrics as sklm

class Metrics(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        self.f1s=sklm.recall_score(targ, predict.round(),average=None)
        return
metrics = Metrics()



callbacks_list = [checkpoint,tb,metrics]
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

#print("coucou")
#print(history.history.keys())
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

#print(y_pred[0],y_test[0])
yy_test = []
yy_pred = []
for i in range(len(y_test)):
    if y_pred[i][0] > 0.5:
        yy_pred.append(1)
    else:
        yy_pred.append(0)
    
    if y_test[i][0] > 0.5:
        yy_test.append(1)
    else:
        yy_test.append(0)
#print("tst",len(yy_test),"pred",len(yy_pred))

# In[17]:

#print("coucou2")
print(confusion_matrix(yy_test, yy_pred))
print("precision : ",precision_score(yy_test, yy_pred, average="macro"))
print("rappel : ",recall_score(yy_test, yy_pred, average="macro"))
print(len(x_test))
