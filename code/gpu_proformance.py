import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import random
import shutil
from matplotlib.image import imread

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import time
import seaborn as sns


import json
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)


import DCGAN as gan

IMG_DIR = './PokemonData'
dirs = os.listdir(IMG_DIR)
print(dirs[:10])

def files_info_extractor(path):
    dir_list = os.listdir(path)
    y = []
    X = []
    for i in range(len(dir_list)):
        files = os.listdir(os.path.join(path,dir_list[i]))
        for j in range(len(files)):
            X.append([files[j],dir_list[i]])
    return X

Total_CLASS_NUMBER = 150
x = files_info_extractor(IMG_DIR)
X = pd.DataFrame(x,columns=["file","label"])
X.sample(10)

def create_train_val_test_folder():
    os.system("mkdir train")
    os.system("mkdir test")
    os.system("mkdir val")
    
def clean_images():
    os.system("rm -r train/* test/* val/*")
    os.system("find train/ -name '*.*' -type f -delete")
    os.system("find val/ -name '*.*' -type f -delete")
    os.system("find test/ -name '*.*' -type f -delete")

def data_extractor(class_pct, train_pct, val_pct, test_pct):
    sub_classes = random.sample(dirs,int(len(dirs)*class_pct))

    for cla in sub_classes:
        os.system('mkdir "train/'+cla+'"')
        os.system('mkdir "test/'+cla+'"')
        os.system('mkdir "val/'+cla+'"')
        os.system("find train/"+cla+" -name '*.*' -type f -delete")
        os.system("find val/"+cla+" -name '*.*' -type f -delete")
        os.system("find test/"+cla+" -name '*.*' -type f -delete")

        temp_files = os.listdir(os.path.join(IMG_DIR,cla))

        files = [f for f in temp_files if isfile(os.path.join(IMG_DIR, cla,f))]
        
        train_files = random.sample(files,int(len(files)*(train_pct+val_pct+test_pct)))
        val_files = random.sample(train_files,int(len(files)*(val_pct+test_pct)))
        test_files = random.sample(val_files,int(len(files)*test_pct))
        train_files = [x for x in train_files if x not in val_files]
        val_files = [x for x in val_files if x not in test_files]
        
        for file in train_files:
            shutil.copy(os.path.join(IMG_DIR,cla,file), os.path.join('train',cla,file))
        for file in val_files:
            shutil.copy(os.path.join(IMG_DIR,cla,file), os.path.join('val',cla,file))
        for file in test_files:
            shutil.copy(os.path.join(IMG_DIR,cla,file), os.path.join('test',cla,file))

SampleSize= 0.2
TrainSize = 0.7
TestSize = 0.2
ValSize= 0.1

ClassNum = int(150 * SampleSize)

image_shape = (256,256,3)
def imageDataGenerator():
    datagen = ImageDataGenerator(rotation_range=20,
                               rescale = 1./255,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='nearest')

    train = datagen.flow_from_directory('train/',target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=15,
                                                class_mode='categorical')
    test = datagen.flow_from_directory('test/',target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=15,
                                                class_mode='categorical')
    val = datagen.flow_from_directory('val/',target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=15,
                                                class_mode='categorical')
    return train, test, val
train, test, val = imageDataGenerator()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Lambda
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

EPOCH = 100

def build_baseline():
    model = Sequential()

    model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',input_shape=image_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=image_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=image_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(64,activation='relu'))

    model.add(Dense(16,activation='relu'))

    model.add(Dense(ClassNum,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return model


baseline_model = build_baseline()
history = baseline_model.fit(train, epochs=EPOCH, validation_data=val, verbose=1)

