import random
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


def getName(filePath):
    return filePath.split('\\')[-1]


def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    data['Center'] = data['Center'].apply(getName)
    data['Right'] = data['Right'].apply(getName)
    data['Left'] = data['Left'].apply(getName)
    print("Total Images Imported:", data.shape[0])
    return data


def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 3000
    hist, bins = np.histogram(data['Steering'], nBins)
    center = (bins[:-1] + bins[1:]) * 0.5
    if display:
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.title('Total number of samples per steering angle')
        plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if bins[j] <= data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)

    print('Removed Images:', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace=True)
    print('Remaining Images:', len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.title('Chosen number of samples per steering angle')
        plt.show()

    return data


def loadData(path, data):
    imagesPath = []
    steering = []
    for i in range(len(data)):
        choice = np.random.choice(3)
        if choice == 0:
            indexedData = data.iloc[i]
            imagesPath.append(os.path.join(path, 'IMG', indexedData[0]))
            steering.append(float(indexedData[3]))
        elif choice == 1:
            indexedData = data.iloc[i]
            imagesPath.append(os.path.join(path, 'IMG', indexedData[1]))
            steering.append(float(indexedData[3] + 0.2))
        elif choice == 2:
            indexedData = data.iloc[i]
            imagesPath.append(os.path.join(path, 'IMG', indexedData[2]))
            steering.append(float(indexedData[3] - 0.2))

    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)

    return imagesPath, steering


def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    ## PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
        img = pan.augment_image(img)
    ## ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    ## BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)
    ## FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering


def preProcess(img):
    img = img[60:135, :, :]

    img = cv2.GaussianBlur(img, (7, 7), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    low_yellow = np.array([22, 40, 0])  # [22, 40, 0] [30, 36, 120]
    up_yellow = np.array([45, 255, 255])  # [45, 255, 255] [45, 45, 160]

    mask = cv2.inRange(hsv, low_yellow, up_yellow)

    edges = cv2.Canny(mask, 50, 120)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, maxLineGap=2, minLineLength=5)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img = cv2.resize(img, (200, 66))

    img = img / 255

    return img


def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]

            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield np.asarray(imgBatch), np.asarray(steeringBatch)


def createModel():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu', kernel_regularizer=regularizers.l2(0.03)))
    model.add(Dense(50, activation='elu', kernel_regularizer=regularizers.l2(0.03)))
    model.add(Dense(10, activation='elu', kernel_regularizer=regularizers.l2(0.03)))
    model.add(Dense(1, activation='elu'))

    model.compile(Adam(learning_rate=0.0001), loss='mse')

    return model
