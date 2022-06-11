import numpy as np
import glob
from PIL import Image
import tensorflow as tf
from keras.layers import *
from keras import Model
from keras import regularizers
from keras.callbacks import CSVLogger
import pandas as pd

part_names = ['Training', 'Validation', 'Test']
outputs = []
inputs = []
for i in range(3):
    dir = './Data/' + str(part_names[i]) + '_Set/'
    reader = pd.read_csv(dir + str(part_names[i]) + '.csv', header=None)
    reader = np.array(reader)
    outputs.extend(reader)
    imdir = dir + part_names[i]
    filelist = glob.glob(imdir + '/*.png')
    images = [np.array(Image.open(fname).resize((256, 128), Image.LANCZOS)) for fname in filelist]
    inputs.extend(images)

inputs = np.asarray(inputs)
outputs = np.array(outputs)

training_inputs = inputs[:1920, :, :, :]
validation_inputs = inputs[1920:2560, :, :, :]
evaluation_inputs = inputs[2560:, :, :, :]
training_outputs = outputs[:1920, :]
validation_outputs = outputs[1920:2560, :]
evaluation_outputs = outputs[2560:, :]
del inputs, outputs

ENC = tf.keras.models.load_model('./Models/MIMO Encoder Model')
Inp = Input(shape=(128, 256, 3))
Ltn = ENC(Inp)
enc = Model(inputs=Inp, outputs=Ltn)
enc.trainable = True

inp = Input(shape=(128, 256, 3))
ltn = enc(inp)
ltn = Conv2D(20, 10, activation="relu", padding="same", strides=(1, 1), kernel_regularizer=regularizers.l2(0.1))(ltn)
ltn = BatchNormalization()(ltn)

ltn = AveragePooling2D(2, strides=(2, 2))(ltn)
ltn = Dropout(0.4)(ltn)

ltn = Conv2D(40, 10, activation="relu", padding="same", strides=(1, 1), kernel_regularizer=regularizers.l2(0.1))(ltn)
ltn = Dropout(0.4)(ltn)
ltn = BatchNormalization()(ltn)

ltn = AveragePooling2D(2, strides=(2, 2))(ltn)
ltn = GlobalAvgPool2D()(ltn)

ltn = Dense(128, activation="softmax", kernel_regularizer=regularizers.l1_l2(l1=0.05, l2=0.05))(ltn)
out = Dense(46, activation="softmax", kernel_regularizer=regularizers.l1_l2(l1=0.05, l2=0.05))(ltn)

cls = Model(inputs=inp, outputs=out)
cls.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

cls.compile(optimizer=opt, loss=loss)
log_dir = "./Logs/Encoder-Based Classifier Model (Trainable Encoder)/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
directory = './Logs/Encoder-Based Classifier Model (Trainable Encoder)/loss.csv'
csv_logger = CSVLogger(directory, append=True, separator=',')


cls.fit(training_inputs, training_outputs, shuffle=True, epochs=500, batch_size=32,
        callbacks=[tensorboard_callback, csv_logger], validation_data=(validation_inputs, validation_outputs))

cls.save('./Models/Encoder-Based Classifier Model (Trainable Encoder)')
cls.compile(metrics=tf.keras.metrics.CategoricalAccuracy())
cls.evaluate(evaluation_inputs, evaluation_outputs, batch_size=32)
