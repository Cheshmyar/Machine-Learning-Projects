import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras import Model
from keras.callbacks import CSVLogger

labels = pd.read_csv('./data/labels.csv')
Y = labels.iloc[:, -1].values

dataset = pd.read_csv('./data/dataset.csv')
X = dataset.iloc[:, 2:].values

le = LabelEncoder()
for i in range(X.shape[1]):
    X[:, i] = le.fit_transform(X[:, i])
Y = le.fit_transform(Y)

Y = tf.keras.utils.to_categorical(Y, num_classes=4)

ss = StandardScaler()
X = ss.fit_transform(X)

X_train, X_eval, Y_train, Y_eval = train_test_split(X, Y, train_size=0.8)

inp = Input(shape=X_train.shape[1])
ltn = Dense(X_train.shape[1] / 2, activation='relu')(inp)
ltn = Dense(X_train.shape[1] / 8, activation='relu')(ltn)
ltn = Dense(X_train.shape[1] / 32, activation='relu')(ltn)
out = Dense(Y.shape[1], activation='softmax')(ltn)

cls = Model(inputs=inp, outputs=out)

cls.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

cls.compile(optimizer=opt, loss='categorical_crossentropy', metrics=tf.keras.metrics.CategoricalAccuracy())
log_dir = "./Logs/Simple Classifier V1.0/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
directory = './Logs/Simple Classifier V1.0/loss.csv'
csv_logger = CSVLogger(directory, append=True, separator=',')

cls.fit(X_train, Y_train, shuffle=True, epochs=500, batch_size=100, callbacks=[tensorboard_callback, csv_logger],
        validation_split=0.05)

cls.save('./Models/Simple Classifier V1.0/')
results = cls.evaluate(X_eval, Y_eval, verbose=0)
print('Evaluation Loss and Accuracy: ', results)
