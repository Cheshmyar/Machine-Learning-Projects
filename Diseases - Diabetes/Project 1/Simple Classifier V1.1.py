from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import CSVLogger

shape = (128, 128, 3)
ratio = 0.1

class_0_inputs_directory = './data/0/'
filelist = glob.glob(class_0_inputs_directory + '*.jpg')
class_0_inputs = [np.array(Image.open(fname).resize((shape[1], shape[0])).__array__()) for fname in filelist]
class_0_inputs = np.array(class_0_inputs)
class_0_inputs = tf.keras.applications.inception_v3.preprocess_input(class_0_inputs)
class_0_outputs = np.zeros([class_0_inputs.shape[0], 1])

class_1_inputs_directory = './data/1/'
filelist = glob.glob(class_1_inputs_directory + '*.jpg')
class_1_inputs = [np.array(Image.open(fname).resize((shape[1], shape[0])).__array__()) for fname in filelist]
class_1_inputs = np.array(class_1_inputs)
class_1_inputs = tf.keras.applications.inception_v3.preprocess_input(class_1_inputs)
class_1_outputs = np.ones([class_1_inputs.shape[0], 1])

inputs = np.concatenate((class_0_inputs, class_1_inputs), axis=0)
outputs = np.concatenate((class_0_outputs, class_1_outputs), axis=0)
del class_0_inputs, class_1_inputs, class_0_outputs, class_1_outputs
print(inputs.shape)
print(outputs.shape)

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=1-ratio, random_state=42)

base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=shape)
for layer in base_model.layers:
    layer.trainable = False
ltn = base_model.output

ltn = Conv2DTranspose(512, 2)(ltn)
ltn = Dropout(0.5)(ltn)
ltn = BatchNormalization()(ltn)

ltn = GlobalAvgPool2D()(ltn)

ltn = Dense(64, activation="relu")(ltn)
ltn = Dropout(0.5)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Dense(8, activation="relu")(ltn)
ltn = Dropout(0.5)(ltn)
ltn = BatchNormalization()(ltn)

out = Dense(1, activation="sigmoid")(ltn)

cls = Model(inputs=base_model.inputs, outputs=out)
cls.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

cls.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
log_dir = "./Logs/Simple Classifier Model V1.1/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
directory = './Logs/Simple Classifier Model V1.1/loss.csv'
csv_logger = CSVLogger(directory, append=True, separator=',')
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

cls.fit(X_train, y_train, shuffle=True, epochs=100, batch_size=25, callbacks=[tensorboard_callback, csv_logger],
        validation_data=(X_test, y_test))

cls.save('./Models/Simple Classifier Model V1.1/')
results = cls.evaluate(X_test, y_test, verbose=0)
print("Evaluation Loss, Evaluation Accuracy:", results)
