import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras import Model
from keras.callbacks import CSVLogger
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import shap

# Just One-Time Rejection Vs. More Than One-Time Rejection

train_size = 0.8
full_data_directory = './Data/Full Data.xlsx'
shap.initjs()

data = pd.read_excel(full_data_directory, sheet_name=0)
del data["GraftRejectionMoreThanOnce"]
data = np.array(data)
inputs = data[:, :27]
inputs = normalize(inputs, axis=1, norm='l1')
outputs = data[:, 27:]

idx_second_rejection = np.nonzero(outputs[:, 0])[0]
idx_third_rejection = np.nonzero(outputs[:, 1])[0]
idx_fourth_rejection = np.nonzero(outputs[:, 2])[0]
idx_fifth_rejection = np.nonzero(outputs[:, 3])[0]
idx_sixth_rejection = np.nonzero(outputs[:, 4])[0]
idx_first_rejection = [i for i in range(outputs.shape[0]) if i not in idx_second_rejection]

class_5_idx = idx_sixth_rejection
class_4_idx = [i for i in idx_fifth_rejection if i not in idx_sixth_rejection]
class_3_idx = [i for i in idx_fourth_rejection if i not in idx_fifth_rejection]
class_2_idx = [i for i in idx_third_rejection if i not in idx_fourth_rejection]
class_1_idx = [i for i in idx_second_rejection if i not in idx_third_rejection]
class_0_idx = idx_first_rejection

inputs_0 = inputs[class_0_idx, :]
outputs_0 = 0 * np.ones(len(class_0_idx))
inputs_1 = inputs[class_1_idx, :]
outputs_1 = 1 * np.ones(len(class_1_idx))
inputs_2 = inputs[class_2_idx, :]
outputs_2 = 1 * np.ones(len(class_2_idx))
inputs_3 = inputs[class_3_idx, :]
outputs_3 = 1 * np.ones(len(class_3_idx))
inputs_4 = inputs[class_4_idx, :]
outputs_4 = 1 * np.ones(len(class_4_idx))
inputs_5 = inputs[class_5_idx, :]
outputs_5 = 1 * np.ones(len(class_5_idx))

class_0_inputs = inputs_0
class_0_outputs = outputs_0

class_1_inputs = np.concatenate((inputs_1, inputs_2, inputs_3, inputs_4, inputs_5), axis=0)
class_1_outputs = np.concatenate((outputs_1, outputs_2, outputs_3, outputs_4, outputs_5), axis=0)

main_inputs = np.concatenate((class_0_inputs, class_1_inputs), axis=0)
main_outputs = np.concatenate((class_0_outputs, class_1_outputs), axis=0)

main_inputs = normalize(main_inputs, axis=0, norm='l2')

sm = SMOTE(random_state=42)
main_inputs, main_outputs = sm.fit_resample(main_inputs, main_outputs)
X_train, X_eval, Y_train, Y_eval = train_test_split(main_inputs, main_outputs, train_size=train_size, shuffle=True)


inp = Input(shape=X_train.shape[1])
ltn = Dense(X_train.shape[1] / 2, activation='relu')(inp)
ltn = Dropout(0.25)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Dense(X_train.shape[1] / 2, activation='relu')(ltn)
ltn = Dropout(0.25)(ltn)
ltn = Dense(X_train.shape[1] / 4, activation='relu')(ltn)
ltn = Dropout(0.25)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Dense(X_train.shape[1] / 4, activation='relu')(ltn)
ltn = Dropout(0.25)(ltn)
ltn = Dense(X_train.shape[1] / 8, activation='relu')(ltn)
ltn = Dropout(0.25)(ltn)
ltn = BatchNormalization()(ltn)
ltn = Dense(X_train.shape[1] / 8, activation='relu')(ltn)
ltn = Dropout(0.25)(ltn)
out = Dense(1, activation='sigmoid')(ltn)

cls = Model(inputs=inp, outputs=out)
cls.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

cls.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])
log_dir = "./Logs/Main Code V2.7/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
directory = log_dir + 'loss.csv'
csv_logger = CSVLogger(directory, append=True, separator=',')

cls.fit(X_train, Y_train, shuffle=True, epochs=150, batch_size=25, callbacks=[tensorboard_callback, csv_logger])

cls.save('./Models/Main Code V2.7/')
results = cls.evaluate(X_eval, Y_eval, verbose=0)
print('Evaluation Accuracy: ', results[1])
results = cls.evaluate(main_inputs, main_outputs, verbose=0)
print('Full Evaluation Accuracy: ', results[1])

Y_pred = cls.predict(X_eval)
for i in range(Y_pred.shape[0]):
    if Y_pred[i] >= 0.5:
        Y_pred[i] = 1
    else:
        Y_pred[i] = 0

cm = confusion_matrix(Y_eval, Y_pred, normalize='true', labels=[0, 1])
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

Y_pred = cls.predict(main_inputs)
for i in range(Y_pred.shape[0]):
    if Y_pred[i] >= 0.5:
        Y_pred[i] = 1
    else:
        Y_pred[i] = 0

cm = confusion_matrix(main_outputs, Y_pred, normalize='true', labels=[0, 1])
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

df = pd.read_excel(full_data_directory, sheet_name=0)
df = pd.DataFrame(df)
explainer = shap.Explainer(cls, X_train)
shap_values = explainer(X_eval)
shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=df.columns[:27])
