import pandas as pd
import random
from PIL import Image
import glob
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LeakyReLU
import keras.backend as K


def accu(y_true, y_pred):
    y_true_ = tf.argmax(y_true, axis=-1)
    y_pred_ = tf.argmax(y_pred, axis=-1)
    return K.mean(y_true_ == y_pred_)


def add_noise(img):
    vari = 50
    deviation = vari*random.random()
    noise_ = np.random.normal(0, deviation, img.shape)
    img += noise_
    np.clip(img, 0., 255.)
    return img


batch_size = 25
shape = (300, 300, 3)

inputs_directory = './dataset/images/'
outputs = pd.read_csv('./dataset/labels.csv', header=None)
outputs = np.array(outputs)
filelist_right = glob.glob(inputs_directory + '*.jpg')
inputs = [np.array(Image.open(fname).resize((shape[1], shape[0])).__array__()) for fname in filelist_right]
inputs = np.array(inputs)
inputs = tf.keras.applications.inception_v3.preprocess_input(inputs)
outputs = tf.keras.utils.to_categorical(outputs[:, 2], num_classes=5)

inputs, inputs_valid, outputs, outputs_valid = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(preprocessing_function=add_noise)
it_train = datagen.flow(inputs, outputs, batch_size=batch_size)
it_valid = datagen.flow(inputs_valid, outputs_valid, batch_size=batch_size)

base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=shape)
for layer in base_model.layers:
    layer.trainable = False
ltn = base_model.output

ltn = Conv2DTranspose(64, 7)(ltn)
ltn = LeakyReLU()(ltn)
# ltn = Dropout(0.1)(ltn)
# ltn = BatchNormalization()(ltn)
ltn = Conv2D(32, 5)(ltn)
ltn = LeakyReLU()(ltn)
# ltn = Dropout(0.1)(ltn)
# ltn = BatchNormalization()(ltn)
ltn = Conv2D(16, 5)(ltn)
ltn = LeakyReLU()(ltn)
# ltn = Dropout(0.1)(ltn)
# ltn = BatchNormalization()(ltn)
ltn = Conv2D(8, 5)(ltn)
ltn = LeakyReLU()(ltn)
# ltn = Dropout(0.1)(ltn)
# ltn = BatchNormalization()(ltn)

ltn = Flatten()(ltn)

ltn = Dense(32)(ltn)
ltn = LeakyReLU()(ltn)
# ltn = Dropout(0.1)(ltn)
ltn = Dense(8)(ltn)
ltn = LeakyReLU()(ltn)
# ltn = Dropout(0.1)(ltn)

out = Dense(5, activation="softmax")(ltn)

cls = Model(inputs=base_model.inputs, outputs=out)
cls.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

cls.compile(optimizer=opt, loss='categorical_crossentropy', metrics=accu)
log_dir = "./Logs/Augmented Classifier Model (Output 2) V1.4/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
directory = './Logs/Augmented Classifier Model (Output 2) V1.4/loss.csv'
csv_logger = CSVLogger(directory, append=True, separator=',')

cls.fit(it_train, steps_per_epoch=int(inputs.shape[0]/batch_size), shuffle=True, epochs=30, validation_data=it_valid,
        validation_steps=int(inputs_valid.shape[0]/batch_size), callbacks=[tensorboard_callback, csv_logger])

cls.save('./Models/Augmented Classifier Model (Output 2) V1.4/')
predicts = cls.predict(inputs_valid)

trueLabels = np.argmax(outputs_valid, axis=-1)
predLabels = np.argmax(predicts, axis=-1)
acc = np.mean(trueLabels == predLabels)

print("Evaluation Accuracy : " + str(acc*100) + "%")
