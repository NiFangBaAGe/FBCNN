from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)

from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import os
import cv2
import matplotlib.pyplot as plt
def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def read_img(img_path):
    img = cv2.imread(img_path)
    pimg = cv2.resize(img, (img_shape[0], img_shape[1]))
    img = pimg
    pimg = np.expand_dims(pimg, axis=0)
    pimg = pimg.astype('float32') / 255
    pimg = pimg - np.load('ruxian_224_train_mean.npy')
    return img, pimg

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer=None):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        new_model = VGG16(weights='imagenet')
    return new_model

def deprocess_image(x):
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(model, x, category_index, layer_name):
    class_output = model.output[:, category_index]
    convolution_output = model.get_layer(layer_name).output
    grads = K.gradients(class_output, convolution_output)[0]
    gradient_function = K.function([model.input], [convolution_output, grads])

    output, grads_val = gradient_function([x])
    output, grads_val = output[0], grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    cam = cv2.resize(cam, (x.shape[1], x.shape[2]), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    image_rgb = x[0, :]
    image_rgb -= np.min(image_rgb)
    image_rgb = np.minimum(image_rgb, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image_rgb)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def vis_heatmap(img, heatmap):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()

    plt.subplot(221)
    plt.imshow(cv2.resize(img, (img_shape[1], img_shape[0])))
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(heatmap)
    plt.axis('off')

    plt.subplot(212)
    heatmap = cv2.resize(heatmap, (img_shape[1], img_shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img,0.6,heatmap,0.4,0)
    plt.imshow(superimposed_img)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('heatmap.jpg', dpi=600)
    plt.show()
    return superimposed_img

def vis(img, heatmap):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()

    plt.subplot(221)
    plt.imshow(cv2.resize(img, (img_shape[1], img_shape[0])))
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(heatmap)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


model = _
last_conv_layer = _
img, pimg = read_img('')
predictions = model.predict(pimg)
predicted_class = np.argmax(predictions[0])
cam, heatmap = grad_cam(model, pimg, predicted_class, last_conv_layer)
superimposed_img = vis_heatmap(img, heatmap)
