import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import pydotplus

x = pd.read_csv("./data/dataset.csv")
x = x[x.columns[2:]]
y = pd.read_csv("./data/labels.csv")
y = y[y.columns[2:]]

x = x.apply(LabelEncoder().fit_transform)
y = y.apply(LabelEncoder().fit_transform)

dtc = tree.DecisionTreeClassifier(max_depth=3)
dtc.fit(x, y)

dot_data = tree.export_graphviz(dtc, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png('graph.png')
