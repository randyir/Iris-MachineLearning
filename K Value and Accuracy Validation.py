import seaborn as a 
import matplotlib.pyplot as b 
import pandas as c 
import numpy as d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

irisDataset = c.read_csv('iris.csv', header=0)
y = irisDataset.iloc[:, 4].values
x = irisDataset.iloc[:, :-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

#Menghitung Akurasi Model
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(classification_report(y_test, y_pred))

#Menentukan Nilai K
error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(d.mean(pred_i != y_test))
    
b.figure(figsize=(12, 6))
b.plot(range(1, 40), error, color = 'orange', marker = 'o', markerfacecolor='orange', markersize='8')
b.title('Error Rate Nilai K')
b.xlabel('Nilai K')
b.ylabel('Error rata-rata')
b.show()