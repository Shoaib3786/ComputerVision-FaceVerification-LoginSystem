""" Goal: """
# Getting Embedded Features using VggNet from Images

""" Libraries """
from keras_vggface.vggface import VGGFace  # model.py change  to tensorflow.keras.utils
from keras.layers import MaxPooling2D, ZeroPadding2D, Convolution2D, Dropout, Flatten, Activation
from keras.models import Model, Sequential


def vgg_model_building():
    """Method-01: Build Vgg model (BASED ON RESNET50)"""
    vgg_model = VGGFace(model='resnet50', weights='vggface', include_top=False,
                        input_shape=(224, 224, 3), pooling='avg')

    ########################################################################

    """Method-02: Vgg model (BASED ON VGG16)"""
    # model = Sequential()
    # model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    # model.add(Convolution2D(64, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(128, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Convolution2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #
    # model.add(Convolution2D(4096, (7, 7), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Convolution2D(4096, (1, 1), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Convolution2D(2622, (1, 1)))
    # model.add(Flatten())
    # model.add(Activation('softmax'))
    #
    # # link to download these weights is given below
    # model.load_weights('vgg_face_weights.h5')
    #
    # vgg_model = Model(inputs=model.layers[0].input,
    #                   outputs=model.layers[-2].output)

    return vgg_model


def embedded_feature(model, img):
    return model.predict(img)

