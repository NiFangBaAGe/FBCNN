from __future__ import print_function
from models import inception_v3, inception_v3_rename, resnet_50, \
    resnet_50_rename, attention_module, inception_resnet_v2, inception_resnet_v2_rename
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Lambda,Reshape,MaxPooling2D
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Concatenate,BatchNormalization,Flatten,AveragePooling2D,GlobalAveragePooling2D
from keras.regularizers import l2
def _outer_product(x):
    return K.batch_dot(x[0], x[1], axes=[1, 1]) / x[0].get_shape().as_list()[1]


def _signed_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-9)


def _l2_normalize(x, axis=-1):
    return K.l2_normalize(x, axis=axis)

class FBCNN():
    def __init__(self, base_model=None, input_shape=None,
                 num_classes=None, bcnn_cfg=None,
                 attention_module=None):
        self.base_model = base_model
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.bcnn_cfg = bcnn_cfg
        self.attention_module = attention_module
        self.basenet1 = None
        self.basenet2 = None
        self.output = None
        self.model = None
        self.model_detector = None
        self.model_extractor = None
        self.get_cfg()

    def build_bilinear_cnn(self, last_conv_layer):
        output_detector = self.basenet1.layers[last_conv_layer].output
        shape_detector = self.basenet1.layers[last_conv_layer].output_shape

        output_extractor = self.basenet2.layers[last_conv_layer].output
        shape_extractor = self.basenet2.layers[last_conv_layer].output_shape

        output_detector = Reshape(
            [shape_detector[1] * shape_detector[2], shape_detector[-1]])(output_detector)
        output_extractor = Reshape(
            [shape_extractor[1] * shape_extractor[2], shape_extractor[-1]])(output_extractor)

        x = Lambda(_outer_product)([output_detector, output_extractor])
        x = Reshape([shape_detector[-1] * shape_extractor[-1]])(x)
        x = Lambda(_signed_sqrt)(x)
        x = Lambda(_l2_normalize)(x)
        output_tensor = Dense(self.num_classes, activation='softmax')(x)
        return output_tensor

    def bulid_bcnn_0(self):
        input_tensor = Input(shape=self.input_shape)
        if self.base_model == 'inception_v3':
            self.basenet1 = inception_v3.InceptionV3(input_shape=self.input_shape, classes=self.num_classes,
                                                     input_tensor=input_tensor, include_top=False, weights='imagenet')
            self.model_detector = self.basenet1
            x = GlobalAveragePooling2D(name='avg_pool')(self.basenet1.output)
            self.output = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        elif self.base_model == 'resnet_50':
            self.basenet1 = resnet_50.ResNet50(include_top=False, input_tensor=input_tensor, weights='imagenet',
                                               input_shape=self.input_shape)
            self.model_detector = self.basenet1
            x = Flatten()(self.basenet1.output)
            self.output = Dense(self.num_classes, activation='softmax')(x)
        elif self.base_model == 'inception_resnet_v2':
            self.basenet1 = inception_resnet_v2.InceptionResNetV2(input_shape=self.input_shape, classes=self.num_classes,
                                                                  include_top=False, weights='imagenet',
                                                                  input_tensor=input_tensor)
            self.model_detector = self.basenet1
            x = GlobalAveragePooling2D(name='avg_pool')(self.basenet1.output)
            self.output = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        self.model = Model(input_tensor, self.output)

    def bulid_bcnn_1(self):
        input_tensor = Input(shape=self.input_shape)
        if self.base_model == 'inception_v3':
            self.basenet1 = inception_v3.InceptionV3(input_shape=self.input_shape, classes=self.num_classes,
                                                     input_tensor=input_tensor, include_top=False, weights='imagenet')
            self.basenet2 = self.basenet1
            self.model_detector = self.basenet1
            self.model_extractor = self.basenet2
            self.output = self.build_bilinear_cnn(last_conv_layer=-1)
        elif self.base_model == 'resnet_50':
            self.basenet1 = resnet_50.ResNet50(include_top=False, weights='imagenet',
                                               input_tensor=input_tensor, input_shape=self.input_shape)
            self.basenet2 = self.basenet1
            self.model_detector = self.basenet1
            self.model_extractor = self.basenet2
            self.output = self.build_bilinear_cnn(last_conv_layer=-2)
        elif self.base_model == 'inception_resnet_v2':
            self.basenet1 = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                  input_shape=self.input_shape,
                                                                  input_tensor=input_tensor)
            self.basenet2 = self.basenet1
            self.model_detector = self.basenet1
            self.model_extractor = self.basenet2
            self.output = self.build_bilinear_cnn(last_conv_layer=-1)
        self.model = Model(input_tensor, self.output)

    def bulid_bcnn_2(self):
        input_tensor = Input(shape=self.input_shape)
        if self.base_model == 'inception_v3':
            self.basenet1 = inception_v3_rename.InceptionV3(input_shape=self.input_shape, classes=self.num_classes, weights='imagenet',
                                                            input_tensor=input_tensor, include_top=False, name='basenet1')
            self.basenet2 = inception_v3_rename.InceptionV3(input_shape=self.input_shape, classes=self.num_classes, weights='imagenet',
                                                            input_tensor=input_tensor, include_top=False, name='basenet2')
            self.model_detector = self.basenet1
            self.model_extractor = self.basenet2
            self.output = self.build_bilinear_cnn(last_conv_layer=-1)
        elif self.base_model == 'resnet_50':
            self.basenet1 = resnet_50_rename.ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape,
                                                      input_tensor=input_tensor,name='basenet1')
            self.basenet2 = resnet_50_rename.ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape,
                                                      input_tensor=input_tensor,name='basenet2')
            self.model_detector = self.basenet1
            self.model_extractor = self.basenet2
            self.output = self.build_bilinear_cnn(last_conv_layer=-2)
        elif self.base_model == 'inception_resnet_v2':
            self.basenet1 = inception_resnet_v2_rename.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                         input_shape=self.input_shape,
                                                                         input_tensor=input_tensor,name='basenet1')
            self.basenet2 = inception_resnet_v2_rename.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                         input_shape=self.input_shape,
                                                                         input_tensor=input_tensor,name='basenet2')
            self.model_detector = self.basenet1
            self.model_extractor = self.basenet2
            self.output = self.build_bilinear_cnn(last_conv_layer=-1)
        self.model = Model(input_tensor, self.output)

    def build_attention(self):
        input_tensor = Input(shape=self.input_shape)
        if self.base_model == 'inception_v3':
            self.model = inception_v3.InceptionV3(input_shape=self.input_shape, classes=self.num_classes, include_top=True,
                                                  weights=None,
                                                  input_tensor=input_tensor, attention_module=self.attention_module)
            self.model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    by_name=True)
        elif self.base_model == 'resnet_50':
            self.model = resnet_50.ResNet50(input_shape=self.input_shape, classes=self.num_classes, include_top=True,
                                            weights=None,
                                            input_tensor=input_tensor, attention_module=self.attention_module)
            self.model.load_weights('resnet_50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    by_name=True)
        elif self.base_model == 'inception_resnet_v2':
            self.model = inception_resnet_v2.InceptionResNetV2(input_shape=self.input_shape, classes=self.num_classes,
                                                               include_top=True, weights=None, input_tensor=input_tensor,
                                                               attention_module=self.attention_module)
            self.model.load_weights('inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    by_name=True)

    def get_cfg(self):
        if self.attention_module:
            self.build_attention()
        elif self.bcnn_cfg == 0:
            self.bulid_bcnn_0()
        elif self.bcnn_cfg == 1:
            self.bulid_bcnn_1()
        elif self.bcnn_cfg == 2:
            self.bulid_bcnn_2()
