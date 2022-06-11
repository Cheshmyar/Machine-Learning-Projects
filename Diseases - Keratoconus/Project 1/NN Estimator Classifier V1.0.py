import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

labels = pd.read_csv('./data/labels.csv')
dataset = pd.read_csv('./data/dataset.csv')
dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
dataset.drop(['En.Anterior.'], axis=1, inplace=True)
dataset.drop(['idEye'], axis=1, inplace=True)
labels.drop(['Unnamed: 0'], axis=1, inplace=True)
labels.drop(['Data.PLOS_One.idEye'], axis=1, inplace=True)

dataset.apply(LabelEncoder().fit_transform)
labels.apply(LabelEncoder().fit_transform)

my_feature_columns = []
for key in dataset.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


def input_fn(features, labels_, training=True, batch_size=500):
    dataset_ = tf.data.Dataset.from_tensor_slices((dict(features), labels_))
    if training:
        dataset_ = dataset_.shuffle(1000).repeat()
    return dataset_.batch(batch_size)


classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,
                                        hidden_units=[1024, 512, 256],
                                        n_classes=5)

print(dataset.head())

classifier.train(input_fn=lambda: input_fn(dataset, labels['clster_labels'], training=True), steps=4901)

eval_result = classifier.evaluate(input_fn=lambda: input_fn(dataset, labels['clster_labels'], training=False))
acc = eval_result['accuracy']*100
print('\nTest set accuracy: {0:0.2f}%\n'.format(acc))
