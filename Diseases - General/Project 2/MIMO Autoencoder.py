import numpy as np
import pandas as pd
import glob
from PIL import Image
import tensorflow as tf
from keras.layers import *
from keras import Model
from keras import regularizers
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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
    images = [np.array(Image.open(fname).resize((128, 64), Image.ANTIALIAS)) for fname in filelist]
    inputs.extend(images)

inputs = np.asarray(inputs)
outputs = np.array(outputs)

inputs, _, outputs, _ = train_test_split(inputs, outputs, test_size=1-0.02, random_state=42)

ae_inputs = []
for i in range(outputs.shape[0]):
    for j in range(outputs.shape[0]):
        if np.sum(np.multiply(outputs[i, :], outputs[i, :])) != 0:
            input_1 = np.expand_dims(inputs[i, :, :, :], axis=3)
            input_2 = np.expand_dims(inputs[j, :, :, :], axis=3)
            ae_inputs.append(np.concatenate((input_1, input_2), axis=3))

ae_inputs = np.array(ae_inputs)
print(ae_inputs.shape)

inp_1 = Input(shape=(64, 128, 3))
inp_2 = Input(shape=(64, 128, 3))

ltn_1 = Conv2D(5, 10, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(0.1))(inp_1)
ltn_1 = Dropout(0.4)(ltn_1)
ltn_1 = BatchNormalization()(ltn_1)
ltn_2 = Conv2D(5, 10, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(0.1))(inp_2)
ltn_2 = Dropout(0.4)(ltn_2)
ltn_2 = BatchNormalization()(ltn_2)

ltn = Concatenate(axis=3)([ltn_1, ltn_2])
ltn = Conv2D(5, 10, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(0.1))(ltn)
ltn = Dropout(0.4)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Conv2D(3, 10, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(0.1))(ltn)
ltn = Dropout(0.4)(ltn)
ltn_main = BatchNormalization()(ltn)

ltn_4 = Conv2D(5, 10, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(0.1))(ltn_main)
ltn_4 = Dropout(0.4)(ltn_4)
ltn_4 = BatchNormalization()(ltn_4)
ltn_5 = Conv2D(5, 10, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(0.1))(ltn_main)
ltn_5 = Dropout(0.4)(ltn_5)
ltn_5 = BatchNormalization()(ltn_5)

ltn_4 = Conv2D(3, 10, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(0.1))(ltn_4)
ltn_4 = Dropout(0.4)(ltn_4)
out_1 = BatchNormalization()(ltn_4)
ltn_5 = Conv2D(3, 10, activation="sigmoid", padding="same", kernel_regularizer=regularizers.l2(0.1))(ltn_5)
ltn_5 = Dropout(0.4)(ltn_5)
out_2 = BatchNormalization()(ltn_5)

encoder = Model(inputs=[inp_1, inp_2], outputs=ltn_main)
autoencoder = Model(inputs=[inp_1, inp_2], outputs=[out_1, out_2])
autoencoder.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

autoencoder.compile(optimizer=opt, loss=loss)
log_dir = "./Logs/RFMID MIMO Autoencoder/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
directory = './Logs/RFMID MIMO Autoencoder/loss.csv'
csv_logger = CSVLogger(directory, append=True, separator=',')

autoencoder.fit([ae_inputs[:, :, :, :, 0], ae_inputs[:, :, :, :, 1]],
                [ae_inputs[:, :, :, :, 0], ae_inputs[:, :, :, :, 1]], shuffle=True, epochs=150, batch_size=4,
                callbacks=[tensorboard_callback, csv_logger])

encoder.save('./Models/RFMID MIMO Autoencoder')
