from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid


def attach_attention_module(net, attention_module):
    if attention_module == 'se_block':  # SE_block
        net = se_block(net, name)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


def se_block(input_feature, name, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    se_feature = GlobalAveragePooling2D(name=name + '_globalaveragepooling2d')(input_feature)
    se_feature = Reshape((1, 1, channel), name=name + 'reshape')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True, name=name + 'dense1',
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel // ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True, name=name + 'dense2',
                       bias_initializer='zeros')(se_feature)
    assert se_feature._keras_shape[1:] == (1, 1, channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2), name=name + 'permute')(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature
