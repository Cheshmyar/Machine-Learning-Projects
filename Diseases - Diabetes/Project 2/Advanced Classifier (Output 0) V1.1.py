import pandas as pd
from PIL import Image
import glob
import random
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import Model
from keras import regularizers
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
import keras.backend as K


def accu(y_true, y_pred):
    y_true_ = tf.argmax(y_true, axis=-1)
    y_pred_ = tf.argmax(y_pred, axis=-1)
    return K.mean(y_true_ == y_pred_)


def add_noise(img):
    vari = 50
    deviation = vari * random.random()
    noise_ = np.random.normal(0, deviation, img.shape)
    img += noise_
    np.clip(img, 0., 255.)
    return img


shape = (224, 224, 3)
indx = 0
BATCH_SIZE = 32

training_outputs = pd.read_csv('./data/Training Labels.csv', header=None)
training_outputs = np.array(training_outputs)
num_classes = np.amax(training_outputs[:, indx]) + 1
training_outputs = tf.keras.utils.to_categorical(training_outputs[:, indx], num_classes=num_classes)

testing_outputs = pd.read_csv('./data/Testing Labels.csv', header=None)
testing_outputs = np.array(testing_outputs)
testing_outputs = tf.keras.utils.to_categorical(testing_outputs[:, indx], num_classes=num_classes)

training_inputs_directory = './data/Training Set/'
filelist_training = glob.glob(training_inputs_directory + '*.jpg')
training_inputs = [np.array(Image.open(fname).resize((shape[1], shape[0])).__array__()) for fname in filelist_training]
training_inputs = np.array(training_inputs)

testing_inputs_directory = './data/Testing Set/'
filelist_testing = glob.glob(testing_inputs_directory + '*.jpg')
testing_inputs = [np.array(Image.open(fname).resize((shape[1], shape[0])).__array__()) for fname in filelist_testing]
testing_inputs = np.array(testing_inputs)

img_gen = ImageDataGenerator(zoom_range=0.1, rotation_range=360, fill_mode='constant', cval=0., horizontal_flip=True,
                             vertical_flip=True, preprocessing_function=add_noise)
training_generator = img_gen.flow(training_inputs, training_outputs, batch_size=BATCH_SIZE)
testing_generator = img_gen.flow(testing_inputs, testing_outputs, batch_size=BATCH_SIZE)

base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=shape)

for layer in base_model.layers:
    layer.trainable = False
ltn = base_model.output

ltn = Conv2DTranspose(60, 2, activation="relu")(ltn)
ltn = Dropout(0.5)(ltn)
ltn = BatchNormalization()(ltn)

ltn = Conv2DTranspose(6, 2)(ltn)
ltn = Dropout(0.5)(ltn)
ltn = BatchNormalization()(ltn)
# , kernel_regularizer=regularizers.l1_l2(l1=0.05, l2=0.05)
ltn = Conv2D(9, 2, activation="relu", padding="same")(ltn)
ltn = Dropout(0.2)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Conv2D(18, 2, activation="relu", padding="same")(ltn)
ltn = Dropout(0.2)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Conv2D(36, 2, activation="relu", padding="same")(ltn)
ltn = Dropout(0.2)(ltn)
ltn = BatchNormalization()(ltn)

ltn = GlobalAvgPool2D()(ltn)

ltn = Dense(18, activation="relu")(ltn)
ltn = Dropout(0.5)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Dense(9, activation="relu")(ltn)
ltn = Dropout(0.5)(ltn)
ltn = BatchNormalization()(ltn)
out = Dense(num_classes, activation="softmax")(ltn)

cls = Model(inputs=base_model.inputs, outputs=out)
cls.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

cls.compile(optimizer=opt, loss='categorical_crossentropy', metrics=tf.keras.metrics.CategoricalAccuracy())
log_dir = "./Logs/Advanced Classifier Model V1.1 (Output 0)/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
directory = './Logs/Advanced Classifier Model V1.1 (Output 0)/loss.csv'
csv_logger = CSVLogger(directory, append=True, separator=',')

cls.fit(training_generator, shuffle=True, epochs=200, callbacks=[tensorboard_callback, csv_logger], validation_data=testing_generator)

cls.save('./Models/Advanced Classifier Model V1.1 (Output 0)/')
results = cls.evaluate(testing_inputs, testing_outputs, verbose=0)
print("Evaluation Loss, Evaluation Accuracy:", results)
