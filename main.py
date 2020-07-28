from __future__ import print_function
import keras
import tensorflow as tf
from load_model import FBCNN
from keras.callbacks import ReduceLROnPlateau


if __name__ == '__main__':
    # bcnn_cfg: 0.base model   1.fast bcnn    2. bcnn
    # attention_module : se_block
    base_model = 'resnet_50'
    num_classes = 8
    bcnn_cfg = 0
    attention_module = None

    optimizer = SGD(lr=0.1)
    ubcnn.model.compile(loss='categorical_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, factor=1 / 10)
    callbacks = [lr_reducer]

    model = FBCNN(base_model=base_model, input_shape=input_shape,
                  num_classes=num_classes, bcnn_cfg=bcnn_cfg, attention_module=attention_module)

    x_train, x_test, y_train, y_test = load_dataset()

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=True,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)

    datagen.fit(x_train)

    ubcnn.model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                              validation_data=(x_test, y_test),
                              epochs=100, verbose=1, workers=4,
                              callbacks=callbacks)

    scores = ubcnn.model.evaluate(x_test, y_test, verbose=1)
