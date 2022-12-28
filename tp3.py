# Load libraries
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Chargement des données :
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
dataset = pd.read_csv("diabetes.csv", header=0, names=col_names)

#Adaptation des données :
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = dataset[feature_cols]
y = dataset.label

# Analyse des données :
print(dataset.describe())

# Matrice de carrélation :
print(dataset.corr())

# Division des données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Création de l'arbre de décision :
clf = DecisionTreeClassifier(criterion = 'gini',splitter="best",max_depth=15,min_samples_split=16, min_samples_leaf=5)

# Entrainement de l'arbre :
clf = clf.fit(X_train,y_train)

# Tester l'arbre :
y_pred = clf.predict(X_test)

# Evaluation de l'arbre :
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualisation de l'arbre :
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,filled=True)