# import keras
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing import image
from keras import optimizers

from vlaues import *
from Cnn_model import CNNmodel
from VGG16_model import VGGmodel
from vgg16_v2_model import VGGV2model
base_dir = 'dataset/'


def get_images(directory):

    # get directory ready
    train_dir = os.path.join(base_dir, 'traindata/')
    test_dir = os.path.join(base_dir, 'testdata/')

    train_men_dir = os.path.join(train_dir, 'men')
    train_women_dir = os.path.join(train_dir, 'women')

    test_men_dir = os.path.join(test_dir, 'men')
    test_women_dir = os.path.join(test_dir, 'women')

    # img_path = os.path.join(test_women_dir, '/00000046.jpg')
    # totalPath = train_women_dir + img_path

    return train_dir, test_dir


def image_ganrator():
    return ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,  # rotation
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,  # apply horizontal_flip
        zoom_range=0.5  # apply zoom
    )


def loading_dataset():
    path_training, path_testing = get_images(base_dir)
    image_gen_train = image_ganrator()
    train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=path_training,
                                                         shuffle=True,
                                                         target_size=(
                                                             IMAGE_HEIGHT, IMAGE_WIDTH),
                                                         class_mode='binary')
    test_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                        directory=path_testing,
                                                        shuffle=True,
                                                        target_size=(
                                                            IMAGE_HEIGHT, IMAGE_WIDTH),
                                                        class_mode='binary')
    return train_data_gen, test_data_gen


def polt_result(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'],
               loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

    # Summarize the model loss

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'],
               loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def trining_model():
    model = VGGV2model()
    tr_gen, tes_gen = loading_dataset()
    # history = model.fit_generator(
    #     tr_gen, epochs=EPOCHS, validation_data=tes_gen)
    # polt_result(history)
    # model.save('./models/VGG_v2_classifor_model.h5')
    return model
