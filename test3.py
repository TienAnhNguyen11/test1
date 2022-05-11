# anh la goodboy
import tensorflow as tf
from tensorflow import keras
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
import os
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Conv2D, MaxPool2D, BatchNormalization, GlobalMaxPool2D, Flatten, Dropout
from sklearn.model_selection import train_test_split


data = "C:\\Users\\DELL\\Downloads\\Train"   # đổi thành đường dẫn ở laptop của em
folders = os.listdir(data)
# print(folders)


image_names = []
data_labels = []
data_images = []

size = 64, 64

for folder in folders:
    for file in os.listdir(os.path.join(data, folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data, folder, file))
            data_labels.append(folder)
            img = cv2.imread(os.path.join(data, folder, file))
            im = cv2.resize(img, size)
            data_images.append(im)
        else:
            continue

data = np.array(data_images)
data.shape

data = data.astype('float32') / 255.0
label_dummies = pandas.get_dummies(data_labels)
labels = label_dummies.values.argmax(1)

pandas.unique(data_labels)
pandas.unique(labels)

# call me daddy
union_list = list(zip(data, labels))
random.shuffle(union_list)
train, labels = zip(*union_list)

# chia du lieu thanh train va test
x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=1)

x_train = np.array(x_train)
y_train = np.array(y_train).reshape((-1, 1))
x_test = np.array(x_test)
y_test = np.array(y_test).reshape((-1, 1))


# xay dung model
model = Sequential()
model.add(InputLayer(input_shape=(64, 64, 3,)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(GlobalMaxPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(5, activation='softmax'))
model.add(Flatten())


# compile model
# nhan kem rieng tai nha
# 0328874444
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train, batch_size=8, epochs=20)

# du doan
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
