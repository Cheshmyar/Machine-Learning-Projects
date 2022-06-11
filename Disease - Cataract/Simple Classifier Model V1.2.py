import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras.applications.efficientnet as efn
from tqdm import tqdm

SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
IMG_HEIGHT = 192
IMG_WIDTH = 256

IMG_ROOT = '../data/'
IMG_DIR = [IMG_ROOT + '1_normal',
           IMG_ROOT + '2_cataract',
           IMG_ROOT + '2_glaucoma',
           IMG_ROOT + '3_retina_disease']

OCU_IMG_ROOT = 'D:\\Amir Rahdar\\Codes\\Dis - General\\Ocular Disease Recognition (Kaggle)\\Data\\ODIR-5K\\ODIR-5K\\Training Images\\'
ocu_df = pd.read_excel('D:\\Amir Rahdar\\Codes\\Dis - General\\Ocular Disease Recognition (Kaggle)\\Data\\ODIR-5K\\ODIR-5K\\data.xlsx')


def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)

cat_df = pd.DataFrame(0,
                      columns=['paths',
                               'cataract'],
                      index=range(601))

filepaths = glob.glob(IMG_ROOT + '*/*')

for i, filepath in enumerate(filepaths):
    filepath = os.path.split(filepath)
    cat_df.iloc[i, 0] = filepath[0] + '/' + filepath[1]

    if filepath[0] == IMG_DIR[0]:
        cat_df.iloc[i, 1] = 0
    elif filepath[0] == IMG_DIR[1]:
        cat_df.iloc[i, 1] = 1
    elif filepath[0] == IMG_DIR[2]:
        cat_df.iloc[i, 1] = 2
    elif filepath[0] == IMG_DIR[3]:
        cat_df.iloc[i, 1] = 3

cat_df = cat_df.query('0 <= cataract < 2')
cat_df

print('Number of normal and cataract images')
print(cat_df['cataract'].value_counts())

ocu_df.head()


def has_cataract_mentioned(text):
    if 'cataract' in text:
        return 1
    else:
        return 0


ocu_df['left_eye_cataract'] = ocu_df['Left-Diagnostic Keywords'] \
    .apply(lambda x: has_cataract_mentioned(x))
ocu_df['right_eye_cataract'] = ocu_df['Right-Diagnostic Keywords'] \
    .apply(lambda x: has_cataract_mentioned(x))

le_df = ocu_df.loc[:, ['Left-Fundus', 'left_eye_cataract']] \
    .rename(columns={'left_eye_cataract': 'cataract'})
le_df['paths'] = OCU_IMG_ROOT + le_df['Left-Fundus']
le_df = le_df.drop('Left-Fundus', axis=1)

re_df = ocu_df.loc[:, ['Right-Fundus', 'right_eye_cataract']] \
    .rename(columns={'right_eye_cataract': 'cataract'})
re_df['paths'] = OCU_IMG_ROOT + re_df['Right-Fundus']
re_df = re_df.drop('Right-Fundus', axis=1)

le_df.head()

re_df.head()

print('Number of left eye images')
print(le_df['cataract'].value_counts())
print('\nNumber of right eye images')
print(re_df['cataract'].value_counts())


def downsample(df):
    df = pd.concat([
        df.query('cataract==1'),
        df.query('cataract==0').sample(sum(df['cataract']),
                                       random_state=SEED)
    ])
    return df


le_df = downsample(le_df)
re_df = downsample(re_df)

print('Number of left eye images')
print(le_df['cataract'].value_counts())
print('\nNumber of right eye images')
print(re_df['cataract'].value_counts())

ocu_df = pd.concat([le_df, re_df])
ocu_df.head()

df = pd.concat([cat_df, ocu_df], ignore_index=True)
df

train_df, test_df = train_test_split(df,
                                     test_size=0.2,
                                     random_state=SEED,
                                     stratify=df['cataract'])

train_df, val_df = train_test_split(train_df,
                                    test_size=0.15,
                                    random_state=SEED,
                                    stratify=train_df['cataract'])


def create_datasets(df, img_width, img_height):
    imgs = []
    for path in tqdm(df['paths']):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_width, img_height))
        imgs.append(img)

    imgs = np.array(imgs, dtype='float32')
    df = pd.get_dummies(df['cataract'])
    return imgs, df


train_imgs, train_df = create_datasets(train_df, IMG_WIDTH, IMG_HEIGHT)
val_imgs, val_df = create_datasets(val_df, IMG_WIDTH, IMG_HEIGHT)
test_imgs, test_df = create_datasets(test_df, IMG_WIDTH, IMG_HEIGHT)

train_imgs = train_imgs / 255.0
val_imgs = val_imgs / 255.0
test_imgs = test_imgs / 255.0

f, ax = plt.subplots(5, 5, figsize=(15, 15))
norm_list = list(train_df[0][:25])
for i, img in enumerate(train_imgs[:25]):
    ax[i // 5, i % 5].imshow(img)
    ax[i // 5, i % 5].axis('off')
    if norm_list[i] == 1:
        ax[i // 5, i % 5].set_title('TrainData: Normal')
    else:
        ax[i // 5, i % 5].set_title('TrainData: Cataract')
plt.show()

f, ax = plt.subplots(5, 5, figsize=(15, 15))
norm_list = list(test_df[0][:25])
for i, img in enumerate(test_imgs[:25]):
    ax[i // 5, i % 5].imshow(img)
    ax[i // 5, i % 5].axis('off')
    if norm_list[i] == 1:
        ax[i // 5, i % 5].set_title('TestData: Normal')
    else:
        ax[i // 5, i % 5].set_title('TestData: Cataract')
plt.show()


class Mish(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def mish(x):
    return tf.keras.layers.Lambda(lambda x: x * K.tanh(K.softplus(x)))(x)


tf.keras.utils.get_custom_objects().update({'mish': Activation(mish)})

input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

model = Sequential()
model.add(Conv2D(16, kernel_size=3, padding='same',
                 input_shape=input_shape, activation='mish'))
model.add(Conv2D(16, kernel_size=3, padding='same', activation='mish'))
model.add(BatchNormalization())
model.add(MaxPool2D(3))
model.add(Dropout(0.3))
model.add(Conv2D(16, kernel_size=3, padding='same', activation='mish'))
model.add(Conv2D(16, kernel_size=3, padding='same', activation='mish'))
model.add(BatchNormalization())
model.add(MaxPool2D(3))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

generator = ImageDataGenerator(horizontal_flip=True,
                               height_shift_range=0.1,
                               fill_mode='reflect')

es_callback = tf.keras.callbacks.EarlyStopping(patience=20,
                                               verbose=1,
                                               restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, verbose=1)

history = model.fit(generator.flow(train_imgs,
                                   train_df,
                                   batch_size=BATCH_SIZE),
                    epochs=EPOCHS,
                    steps_per_epoch=len(train_imgs) / BATCH_SIZE,
                    callbacks=[es_callback, reduce_lr],
                    validation_data=(val_imgs, val_df))

pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.show()

model.evaluate(test_imgs, test_df)


def build_model(img_height, img_width, n):
    inp = Input(shape=(img_height, img_width, n))
    efnet = efn.EfficientNetB0(
        input_shape=(img_height, img_width, n),
        weights='imagenet',
        include_top=False
    )
    x = efnet(inp)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.000003)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model


model = build_model(IMG_HEIGHT, IMG_WIDTH, 3)
model.summary()

generator = ImageDataGenerator(horizontal_flip=True,
                               height_shift_range=0.1,
                               fill_mode='reflect')

es_callback = tf.keras.callbacks.EarlyStopping(patience=20,
                                               verbose=1,
                                               restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, verbose=1)

history = model.fit(generator.flow(train_imgs,
                                   train_df,
                                   batch_size=BATCH_SIZE),
                    epochs=EPOCHS,
                    steps_per_epoch=len(train_imgs) / BATCH_SIZE,
                    callbacks=[es_callback, reduce_lr],
                    validation_data=(val_imgs, val_df))

pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.show()

model.evaluate(test_imgs, test_df)
