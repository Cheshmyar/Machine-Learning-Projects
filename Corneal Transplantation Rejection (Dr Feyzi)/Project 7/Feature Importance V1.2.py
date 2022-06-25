import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

train_size = 0.8
full_data_directory = './Data/Full Data.xlsx'

data = pd.read_excel(full_data_directory, sheet_name=0)
data = np.array(data)
inputs = data[:, :28]
inputs = np.delete(inputs, 17, 1)
inputs = normalize(inputs, axis=1, norm='l1')
outputs = data[:, 28:]

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

class_0_inputs = inputs[class_0_idx, :]
class_0_outputs = 0 * np.ones(len(class_0_idx))
class_1_inputs = inputs[class_1_idx, :]
class_1_outputs = 1 * np.ones(len(class_1_idx))
class_2_inputs = inputs[class_2_idx, :]
class_2_outputs = 2 * np.ones(len(class_2_idx))
class_3_inputs = inputs[class_3_idx, :]
class_3_outputs = 3 * np.ones(len(class_3_idx))
class_4_inputs = inputs[class_4_idx, :]
class_4_outputs = 4 * np.ones(len(class_4_idx))
class_5_inputs = inputs[class_5_idx, :]
class_5_outputs = 5 * np.ones(len(class_5_idx))

main_inputs = np.concatenate((class_0_inputs, class_1_inputs, class_2_inputs, class_3_inputs, class_4_inputs,
                              class_5_inputs), axis=0)
main_outputs = np.concatenate((class_0_outputs, class_1_outputs, class_2_outputs, class_3_outputs, class_4_outputs,
                               class_5_outputs), axis=0)

ss = StandardScaler()
main_inputs = ss.fit_transform(main_inputs)

df = pd.read_excel(full_data_directory, sheet_name=0)
df = pd.DataFrame(df)
del df["GraftRejection"]

model = XGBClassifier()
model.fit(main_inputs, main_outputs)

importances = pd.DataFrame(data={
    'Attribute': df.columns[:27],
    'Importance': model.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)
importances.to_excel('./Excel Files/XGB.xlsx')

plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature Importances, Obtained From Coefficients Using XGBoost', size=20)
plt.xticks(rotation='vertical')
plt.show()

model = LogisticRegression()
model.fit(main_inputs, main_outputs)
importances = pd.DataFrame(data={
    'Attribute': df.columns[:27],
    'Importance': model.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)
importances.to_excel('./Excel Files/LR.xlsx')

plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature Importances, Obtained From Coefficients Using Logistic Regression', size=20)
plt.xticks(rotation='vertical')
plt.show()

pca = PCA().fit(main_inputs)

loadings = pd.DataFrame(
    data=pca.components_.T * np.sqrt(pca.explained_variance_),
    columns=[f'PC{i}' for i in range(1, 28)],
    index=df.columns[:27]
)
loadings.head()

pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
pc1_loadings = pc1_loadings.reset_index()
pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']
pc1_loadings.to_excel('./Excel Files/PCA.xlsx')

plt.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
plt.title('PCA Loading Scores (First Principal Component)', size=20)
plt.xticks(rotation='vertical')
plt.show()
