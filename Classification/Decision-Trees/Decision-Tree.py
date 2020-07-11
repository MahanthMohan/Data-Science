import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn import model_selection

my_data = pd.read_csv('Dataset/drug200.csv')
my_data[0:5]

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
X[0:5]

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

X[0:5]

y = my_data['Drug']
y[0:5]

X_trainset, X_testset, y_trainset, y_testset = model_selection.train_test_split(X, y, test_size=0.3, random_state=3)
drugTree = tree.DecisionTreeClassifier(crtierion='entropy', max_depth = 4)
drugTree.fit(X_trainset, y_trainset)

pred = drugTree.predict(X_testset)
print(pred[0:5])
print(y_testset[0:5])