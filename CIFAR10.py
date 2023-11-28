import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Conv2D,Flatten,Dropout,BatchNormalization,MaxPooling2D
from keras.datasets import cifar10
import numpy as np
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

(X_train,y_train),(X_test,y_test)=cifar10.load_data()

X_train=X_train/255.0
X_test=X_test/255.0
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)


input_layer= keras.Input(shape=X_train.shape[1:])

model=Conv2D(32,(3,3),activation='relu')(input_layer)
model=Conv2D(32,(3,3),activation='relu')(model)
model=MaxPooling2D(pool_size=(2,2))(model)
model=BatchNormalization()(model)

model=Conv2D(64,(3,3),activation='relu',padding='same')(model)
model=BatchNormalization()(model)
model=MaxPooling2D(pool_size=(2,2))(model)
model=Dropout(0.4)(model)

model=Conv2D(256,(5,5),activation='relu',padding='same')(model)
model=MaxPooling2D(pool_size=(2,2))(model)
model=BatchNormalization()(model)
model=Dropout(0.4)(model)

model=Conv2D(64,(4,4),activation='relu',padding='same')(model)
model=MaxPooling2D(pool_size=(2,2))(model)
model=BatchNormalization()(model)

model=Flatten()(model)
model=Dropout(0.4)(model)

model=Dense(256,activation='relu')(model)
model=BatchNormalization()(model)
model=Dropout(0.4)(model)

model=Dense(10,activation='softmax')(model)

model=keras.Model(inputs=input_layer,outputs=model)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# model.summary()

with tf.device('/GPU:0'):
    model.fit(X_train,y_train,batch_size=75,epochs=20,validation_split=0.1)

model.save("CIFAR10.model")
