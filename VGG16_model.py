from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from vlaues import *


def VGGmodel():
    '''
    Using transfer learning for VGG16 model
    :return: model
    '''
    pre_trained_model = applications.VGG16(input_shape=(
        IMAGE_WIDTH, IMAGE_HEIGHT, 3), include_top=False, weights="imagenet")

    pre_trained_model.trainable = False

    last_layer = pre_trained_model.get_layer('block5_pool')
    last_output = last_layer.output
    x = Flatten()(last_output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    vggmodel = Model(pre_trained_model.input, x)

    vggmodel.compile(loss='binary_crossentropy',
                     optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                     metrics=['accuracy'])
    return vggmodel


VGGmodel()
