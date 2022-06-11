import pandas as pd
from PIL import Image
import glob
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import Model
from keras import regularizers
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split

inputs_directory = './Data/preprocessed_images/'
outputs = pd.read_csv('./Data/full_df.csv', header=None)
outputs = np.array(outputs)
filelist_right = glob.glob(inputs_directory + '*right.jpg')
images_right = [np.array(Image.open(fname).resize((128, 64)).__array__()) for fname in filelist_right]
images_right = np.array(images_right)
filelist_left = glob.glob(inputs_directory + '*left.jpg')
images_left = [np.array(Image.open(fname).resize((128, 64)).__array__()) for fname in filelist_left]
images_left = np.array(images_left)
inputs = np.concatenate((images_right, images_left), axis=0)
del images_right, images_left

inputs, inputs_evl, outputs, outputs_evl = train_test_split(inputs, outputs, test_size=0.5, random_state=42,
                                                            shuffle=True)

inp = Input(shape=(64, 128, 3))

ltn = Conv2D(30, 5, activation="relu", padding="same")(inp)
ltn = Dropout(0.1)(ltn)
# ltn = BatchNormalization()(ltn)
ltn = AveragePooling2D(2, strides=(2, 2))(ltn)
ltn = Dropout(0.1)(ltn)

ltn = Conv2D(60, 5, activation="relu", padding="same")(ltn)
ltn = Dropout(0.1)(ltn)
# ltn = BatchNormalization()(ltn)
ltn = AveragePooling2D(2, strides=(2, 2))(ltn)
ltn = Dropout(0.1)(ltn)

ltn = Conv2D(90, 5, activation="relu", padding="same")(ltn)
ltn = Dropout(0.1)(ltn)
# ltn = BatchNormalization()(ltn)
ltn = AveragePooling2D(2, strides=(2, 2))(ltn)
ltn = Dropout(0.5)(ltn)

ltn = GlobalAvgPool2D()(ltn)
ltn = Dense(45, activation="sigmoid")(ltn)
ltn = Dropout(0.1)(ltn)
out = Dense(8, activation="sigmoid")(ltn)

cls = Model(inputs=inp, outputs=out)
cls.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

cls.compile(optimizer=opt, loss='mean_squared_error', metrics=tf.keras.metrics.BinaryAccuracy(threshold=0.5))
log_dir = "./Logs/MIMO Encoder-Based Classifier Model/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
directory = './Logs/MIMO Encoder-Based Classifier Model/loss.csv'
csv_logger = CSVLogger(directory, append=True, separator=',')

cls.fit(inputs, outputs, shuffle=True, epochs=500, batch_size=128, callbacks=[tensorboard_callback, csv_logger],
        validation_split=0.05)

cls.save('./Models/MIMO Encoder-Based Classifier Model')
_, categorical_acc = cls.evaluate(inputs_evl, outputs_evl)
print(f"Accuracy on the evaluation set: {round(categorical_acc * 100, 2)}%.")
