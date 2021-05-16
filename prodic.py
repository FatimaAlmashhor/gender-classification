import tensorflow as tf
from vlaues import *
import matplotlib.pyplot as plt
import os
import numpy as np


def prodic(model):

    dire = "dataset/validatedata/"
    path = os.listdir(os.path.join(dire))
    # path = input("Where is your image path? \n")
    No_img = np.random.randint(len(path))
    image = tf.io.read_file(dire + path[No_img])
    image = tf.image.decode_jpeg(image, channels=3)
    plt.imshow(image)
    image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    image /= 255.0
    image_x = tf.expand_dims(image, axis=0)

    plt.show()
    # print()
    # path_model = os.path.dirname(
    #     __file__)
    # print(path_model)
    # model = tf.keras.models.load_model(
    # '.\VGG_classifor_model.hdf5')

    y = model.predict(image_x)
    index = 0
    if y > 0.5:
        index = 1
    else:
        index = 0
    print("This image shows a " + IDX_TO_LABELS[index])


# prodic()
