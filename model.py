import csv
import keras
import cv2
import numpy as np
from keras.layers import Input, Lambda, Dense, Flatten, Cropping2D, Dropout
from keras.models import Sequential
from keras.layers.pooling import MaxPool2D
from keras.layers.convolutional import Convolution2D
import tensorflow as tf
from scipy import ndimage

from pathlib import Path
home = str(Path.home())

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
#Setting the correction factor for the center, left, and right images.
correction = [0,0.1,-0.1]

#Iterating through each line of the csv file to extract center, left, and right images with steering measurement.
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/'+filename
        image = ndimage.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement+correction[i])

flip_imgs = []

#flipping images to counter the left turns.
for image in images:
    flip_imgs.append(image)
    flip_imgs.append(cv2.flip(image,1))

flip_measurements = []

#flipping the measurements for corresponding images.
for measurement in measurements:
    flip_measurements.append(measurement)
    flip_measurements.append(measurement * -1.0)

#Setting training input
X_train = np.array(flip_imgs)

#Setting the training output.
y_train = np.array(flip_measurements)

#Setting the model to be sequential. 
model = Sequential()

#using Lamba layer to normalize the images.
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

#Cropping top and bottom of the images.
model.add(Cropping2D(cropping=((70,25),(0,0))))

#convoluting with 5x5 filter. 
model.add(Convolution2D(6,5,5,activation='relu'))

#maxpooling the outputs
model.add(MaxPool2D())

#convoluting with 5x5 filter. 
model.add(Convolution2D(16,5,5,activation='relu'))

#maxpooling the outputs
model.add(MaxPool2D())

#flattening the output of the max pooling 
model.add(Flatten())

# 120 fully connected layer
model.add(Dense(120))

# added dropout layer 
model.add(Dropout(0.5))

# 64 fully connected layer
model.add(Dense(64))

# 1 fully connected layer
model.add(Dense(1))

# setting adam optimizer and regression output for the model
model.compile(optimizer='Adam', loss='mse')

# setting training input and output, shuffling the data, validation split, and number of epochs to run.
model.fit(X_train, y_train,validation_split=0.2, shuffle=True, epochs=3)

# saving the weights/model.


# displaying architecture.
model.summary()
