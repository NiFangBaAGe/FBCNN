from keras.applications.vgg16 import VGG16
import keras
import tensorflow as tf
import os
import cv2
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.models import Model
from keras.layers import Input
from keras import backend as K
import matplotlib.pyplot as plt
from  PIL import Image
from keras.applications.vgg16 import preprocess_input

def read_img(img_path, size):
    img = cv2.imread(img_path)
    pimg = cv2.resize(img, (img_shape[0], img_shape[1]))
    pimg = np.expand_dims(pimg, axis=0)
    pimg = pimg.astype('float32') / 255

    return img, pimg

def conv_output(model, layer_name, img):
    input_img = model.layers[1].input
    try:
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    intermediate_output = intermediate_layer_model.predict(img)

    return intermediate_output[0]

def vis_conv_output(img, conv_output):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()

    plt.subplot(221)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(conv_output[:,:,0])
    plt.axis('off')

    plt.tight_layout()
    plt.show()

from load_model import FBCNN
import matplotlib
img_shape = (224,224,3)
if __name__ == '__main__':
    model = _
    last_conv_layer = _
    img, pimg = read_img('', (img_shape[0], img_shape[1]))
    output = conv_output(model, last_conv_layer, pimg)
    vis_conv_output(img, output)
    plt.imsave('conv_output.jpg', output[:, :, 0])

