# Customer Churn Prediction

import dataprep
from dataprep.Package import Package
import pickle
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder


with Package.open_package('CATelcoCustomerChurnTrainingSample.dprep') as pkg:
    df = pkg.dataflows[0].get_dataframe()

print(df.columns)

# Plocka ut kolumner av typen category eller string då vi inte kan köra ML på Strings
columns_to_encode = list(df.select_dtypes(include=['category','object']))
# Skapa upp nya kolumner med värdet 1 om de är satta för varje kategoris värde och slå ihop med dataframe
for column_to_encode in columns_to_encode:
    # Skapa ny kolumn för varje värde en viss kolumn kan ha - värdet blir kolumnnamn
    # Se: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
    dummies = pd.get_dummies(df[column_to_encode])
    # Spara kolumnnamnen
    one_hot_col_names = []
    for col_name in list(dummies.columns):
        # Lägg till ursprungskolumnens namn följt av värdet på kategorin
        one_hot_col_names.append(column_to_encode + '_' + col_name)
    # Tilldela nya kolumnnamn
    dummies.columns = one_hot_col_names
    # Whether to drop labels from the index (0 / ‘index’) or columns (1 / ‘columns’).
    # Se: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html
    df = df.drop(column_to_encode, axis=1)
    #Slå ihop ursprungaliga df (med borttagna kolumner) med nya framtagna kolumner
    df = df.join(dummies)

print(df.columns)

# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
model = GaussianNB()

train, test = train_test_split(df, test_size = 0.3)

# Plocka ut utfall
trainY = train['churn'].values
# len(target[target > 0]) säger hur mycket churn vi har i antal
# Tag bort kolumnen för utfall
trainX = train.drop('churn', 1)

trainX = trainX.values
# train.shape - för att se att vi har 7000 rader och 86 kolumner med värden
model.fit(trainX, trainY)

# utfall testdata
testY = test['churn'].values
# Tag bort kolumn för utfall för testdata
testX = test.drop('churn', 1)
# Gör predict (Score) på testdata på vår tränade modell
predicted = model.predict(testX)
# Spara undan sannolikhet för varje prediciton
predicted_prob = model.predict_proba(testX)
# Hur bra var Naive Bayes modellen för testdata?
print("Naive Bayes Classification Accuracy", accuracy_score(testY, predicted))


"""dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(trainX, trainY)
predicted = dt.predict(testX)
predicted_prob_dt = dt.predict_proba(testX)
fprdt, tprdt, _ = roc_curve(testY, predicted_prob_dt[:, 1])
roc_auc_dt = roc_auc_score(testY, predicted_prob_dt[:, 1])"""
#roc_auc_score(expected, )


# TASK: Rita ROC curve
# confidence_nb = predicted_prob.max(axis=1)
fpr, tpr, _ = roc_curve(testY, predicted_prob[:, 1])
roc_auc_nb = roc_auc_score(testY, predicted_prob[:, 1])
print("AUC NB:" + str(roc_auc_nb))

plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(trainX, trainY)
predicted = dt.predict(testX)
predicted_prob_dt = dt.predict_proba(testX)
fprdt, tprdt, _ = roc_curve(testY, predicted_prob_dt[:, 1])
roc_auc_dt = roc_auc_score(testY, predicted_prob_dt[:, 1])

print("AUC Decision Tree:" + str(roc_auc_dt))

plt.figure()
plt.plot(fprdt, tprdt)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Hur bra var decision tree modellen?
print("Decision Tree Classification Accuracy", accuracy_score(testY, predicted))

# serialize the model on disk in the special 'outputs' folder
"""
print ("Export the model to model.pkl")
f = open('./outputs/model.pkl', 'wb')
pickle.dump(dt, f)
f.close()"""