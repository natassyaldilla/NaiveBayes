import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import class_likelihood_ratios
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn.model_selection as model_selection
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay, classification_report)

pd.options.mode.chained_assignment = None

#MEMBACA DATA
dataframe = pd.read_excel(r"diabetes.xlsx")


data=dataframe[['Pregnancies','Glucose','BloodPressure',
                'SkinThickness','Insulin','BMI',
                'DiabetesPedigreeFunction','Outcome']]


print("data awal".center(75,"="))
print(data)
print("============================================================")
print()

#PENGECEKAN MISSING VALUE
print("pengecekan missing value".center(75, "="))
print(data.isnull().sum())
print("============================================================")
print()

#GROUPING YANG DIBAGI MENJADI DUA
print("GROUPING VARIABEL".center(75, "="))
X = data.iloc[:, 0:6].values
Y = data.iloc[:, 7].values
print("data variabel".center(75, "="))
print(X)
print()
print("data kelas".center(75, "="))
print(Y)
print("============================================================")
print()

#PEMBAGIAN TRAINING DAN TESTING
print("SPLITTING DATA 20-80".center(75, "="))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("instance variabel data training".center(75,"="))
print(X_train)
print()
print("instance kelas data training".center(75,"="))
print(Y_train)
print()
print("instance variabel data testing".center(75,"="))
print(X_test)
print()
print("instance kelas data testing".center(75,"="))
print(Y_test)
print("============================================================")
print()

#PEMODELAN NAIVE BAYES
print("PEMODELAN DENGAN NAIVE BAYES".center(75,"="))
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test) 
accuracy_nb=round(accuracy_score(Y_test,Y_pred)* 100, 2)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print("instance prediksi naive bayes:")
print(Y_pred)
print()


#PERHITUNGAN CONFUSION MATRIX
cm = confusion_matrix(Y_test, Y_pred)
print('CLASSIFICATION REPORT NAIVE BAYES'.center(75,'='))

#MENDAPAT AKURASI
accuracy = accuracy_score(Y_test, Y_pred)

#MENDAPAT AKURASI
precision = precision_score(Y_test, Y_pred)

#MENAMPILKAN RECISION RECALL F1-SCORE SUPPORT
print(classification_report(Y_test, Y_pred))
    
cm = confusion_matrix(Y_test, Y_pred)
TN = cm[1][1] * 1.0
FN = cm[1][0] * 1.0

TP = cm[0][0] * 1.0
FP = cm[0][1] * 1.0
total = TN + FN + TP + FP
sens = TN / (TN + FP) * 100
spec = TP / (TP + FN) * 100
    
print('Akurasi      : ', accuracy * 100, "%")
print('Sensitivity  : ' + str(sens))
print('Specificity  : ' + str(spec))
print('Precision    : ' + str(precision))
print("============================================================")
print()

#MENAMPILKAN CONFUSION MATRIX
cm_display=ConfusionMatrixDisplay(confusion_matrix=cm)

print('Confusion matrix for Naive Bayes\n',cm)
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("============================================================")
print()

#COBA INPUT
A = int(input("Umur Pasien = "))
print("Isi Jenis kelamin dengan 0 jika Perempuan dan dan 1 jika Laki-Laki")
B = input("Jenis Kelamin Pasien = ")
print("Isi Y jika mengalami dan N jika tidak")
C = input("Apakah pasien mengalami Glucose? = ")
D = input("Apakah pasien mengalami Pregnancies? = ")
E = input("Apakah pasien mengalami Blood Pressure? = ") 
F = input("Apakah pasien mengalami Skin Thickness? = ")
G = input("Apakah pasien mengalami Insulin? = ")
H = input("Apakah pasien mengalami Diabetes Pedigree Function? = ")


if C=="Y":
   C=1
else:
   C=0

if D=="Y":
    D=1
else:
   D=0

if E=="Y":
   E=1
else:
   E=0

if F=="Y":
   F=1
else:
   F=0

if G=="Y":
   G=1
else:
   G=0

if H=="Y":
   H=1
else:
   H=0


Train = [C,D,E,F,G,
         H]
print(Train)

test = pd.DataFrame(Train).T

predtest = gaussian.predict(test)

if predtest==1:
    print("Pasien Positive ")
else:
    print("Pasien Negative ")
