import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from albumentations import *
from PIL import Image
import glob
import random
from keras.layers import *
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import CSVLogger
from keras.layers import LeakyReLU

shape = (200, 300, 3)
BATCH_SIZE = 10
ratio = 0.01


def add_noise(img):
    vari = 50
    deviation = vari*random.random()
    noise_ = np.random.normal(0, deviation, img.shape)
    img += noise_
    np.clip(img, 0., 255.)
    return img


datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                             zoom_range=0.2, preprocessing_function=add_noise)

dataset_directory = 'D:\\Amir Rahdar\\Codes\\My Data\\'
filelist_cataract = glob.glob(dataset_directory + 'Cataract\\*.*')
filelist_healthy = glob.glob(dataset_directory + 'Healthy\\*.*')

original_data_cataract = [np.array(Image.open(fname).convert('RGB').resize((shape[1], shape[0])).__array__()) for fname
                          in filelist_cataract]
original_data_cataract = np.array(original_data_cataract)
original_data_healthy = [np.array(Image.open(fname).convert('RGB').resize((shape[1], shape[0])).__array__()) for fname
                         in filelist_healthy]
original_data_healthy = np.array(original_data_healthy)

original_labels_cataract = np.zeros(original_data_cataract.shape[0])
original_labels_healthy = np.ones(original_data_healthy.shape[0])

input_data = np.concatenate((original_data_cataract, original_data_healthy), axis=0)
output_data = np.concatenate((original_labels_cataract, original_labels_healthy), axis=0)

inputs, inputs_valid, outputs, outputs_valid = train_test_split(input_data, output_data, test_size=1 - ratio,
                                                                random_state=42)

it_train = datagen.flow(inputs, outputs, batch_size=BATCH_SIZE)
it_valid = datagen.flow(inputs_valid, outputs_valid, batch_size=BATCH_SIZE)

base_model = VGG16(include_top=False, weights='imagenet', input_shape=shape)
for layer in base_model.layers:
    layer.trainable = False
ltn = base_model.output

ltn = Conv2DTranspose(128, 7)(ltn)
ltn = LeakyReLU()(ltn)
ltn = Dropout(0.1)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Conv2DTranspose(128, 7)(ltn)
ltn = LeakyReLU()(ltn)
ltn = Dropout(0.1)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Conv2D(32, 3)(ltn)
ltn = LeakyReLU()(ltn)
ltn = Dropout(0.1)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Conv2D(16, 3)(ltn)
ltn = LeakyReLU()(ltn)
ltn = Dropout(0.1)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Conv2D(8, 3)(ltn)
ltn = LeakyReLU()(ltn)
ltn = Dropout(0.1)(ltn)
ltn = BatchNormalization()(ltn)

ltn = GlobalAvgPool2D()(ltn)

ltn = Dense(8)(ltn)
ltn = LeakyReLU()(ltn)
ltn = Dropout(0.1)(ltn)
ltn = Dense(4)(ltn)
ltn = LeakyReLU()(ltn)
ltn = Dropout(0.1)(ltn)
ltn = Dense(8)(ltn)
ltn = LeakyReLU()(ltn)
ltn = Dropout(0.1)(ltn)

out = Dense(1, activation="softmax")(ltn)

cls = Model(inputs=base_model.inputs, outputs=out)
cls.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

cls.compile(optimizer=opt, loss='categorical_crossentropy', metrics=tf.keras.metrics.CategoricalAccuracy())
log_dir = "./Logs/Augmented Classifier Model V1.0 - " + str(int(ratio * 100)) + "%/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
directory = log_dir + "loss.csv"
csv_logger = CSVLogger(directory, append=True, separator=',')

cls.fit(it_train, steps_per_epoch=int(inputs.shape[0]/BATCH_SIZE), shuffle=True, epochs=150, validation_data=it_valid,
        validation_steps=int(inputs_valid.shape[0]/BATCH_SIZE), callbacks=[tensorboard_callback, csv_logger])

cls.save("./Models/Augmented Classifier Model V1.0 - " + str(int(ratio * 100)) + "%/")
results = cls.evaluate(inputs_valid, outputs_valid, verbose=0)
print("Evaluation Loss, Evaluation Accuracy:", results)
