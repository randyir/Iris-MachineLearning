import seaborn as a 
import matplotlib.pyplot as b 
import pandas as c
from pandas.core.indexes.api import Index

irisDataset = c.read_csv('iris.csv')
a.scatterplot(x='sepals-length', y='sepals-width', hue='label', data=irisDataset).set_title('Sebaran Data Iris')
b.figure(1)
b.show()