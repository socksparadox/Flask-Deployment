# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Importing Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Importing Dataset
dataset = pd.read_csv("F:/DATASETS/Wine_Quality_Data.csv")
col_names = dataset.columns

dataset.color.unique()

#Function to convert string to int in color column 
def color_to_int(word):
    word_dict = {'red': 1,'white':0}
    return word_dict[word]

dataset['color'] = dataset['color'].apply(lambda x : color_to_int(x))
dataset.isnull().sum()

#Checking for null values and datatypes of the columns 
dataset.info()

# Splitting data into Features and Target.
X = dataset.drop('quality',axis = 1)
y = dataset['quality']

# Splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# K-Nearest-Neighbors Classifier
knnclassifier = KNeighborsClassifier(n_neighbors=9)
knnclassifier.fit(X_train, y_train)
print("The accuracy of K Nearest Neighbors Classifier is",
      knnclassifier.score(X_train,y_train), knnclassifier.score(X_test,y_test))
knn = [knnclassifier.score(X_train,y_train), knnclassifier.score(X_test,y_test)]

# Support Vector Machines Classifier
svm = SVC(C=1E5)
svm.fit(X_train, y_train)
print("The accuracy of SVM is",
      svm.score(X_train,y_train), svm.score(X_test,y_test))
svm = [svm.score(X_train,y_train), svm.score(X_test,y_test)]

#Decision Tree Classifier Model
dtclassifier = DecisionTreeClassifier(max_depth=7)
dtclassifier.fit(X_train,y_train)
print("The accuracy of Decision Tree Classifier is",
      dtclassifier.score(X_train,y_train),dtclassifier.score(X_test,y_test))
dt = [dtclassifier.score(X_train,y_train),dtclassifier.score(X_test,y_test)]

#Random Forest Classifier Model
rfclassifier = RandomForestClassifier(max_depth = 7)
rfclassifier.fit(X_train, y_train)
print("The accuracy of random forest Classifier is",
      rfclassifier.score(X_train,y_train), rfclassifier.score(X_test,y_test))
rf = [rfclassifier.score(X_train,y_train), rfclassifier.score(X_test,y_test)]

#Gradient Boosting Classifier Model
gbclassifier = GradientBoostingClassifier()
gbclassifier.fit(X_train,y_train)
print("The accuracy of Gradient Boosting Classifier is",
      gbclassifier.score(X_train,y_train),gbclassifier.score(X_test,y_test))
gb = [gbclassifier.score(X_train,y_train),gbclassifier.score(X_test,y_test)]

#Results table for comparison of accuracies
results1 = pd.DataFrame(data=[knn,svm,dt,rf,gb],
                        columns = ['Training Accuracy ', 'Testing Accuracy '],
                        index = ['K Nearest Neighbors', 'Support Vector Machines',
                                 'Decision Tree', 'Random Forest', 'Gradient Boost'])

#Creation of pickle file.
pickle.dump(rfclassifier, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
