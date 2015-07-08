# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:17:46 2015

@author: Eku
"""

#pyplot is used to actually plt a chart
#datasets are used as the sample dataset

import matplotlib.pyplot as pit
from sklearn import datasets
from sklearn import svm

#loading the digit dataset
digits = datasets.load_digits()

#specifying the classifier
clf = svm.SVC(gamma = 0.01, C = 100)

#print(len(digits.data))

#specifying the data to be used for training purpose
x,y = digits.data[:-1],digits.target[:-1]

#training the svm
clf.fit(x,y)

#Let's test the prediction now
print("Prediction : ", clf.predict(digits.data[-1]))

#showing results, image of the number to be tested
pit.imshow(digits.images[-1], cmap=pit.cm.gray_r, interpolation = "nearest")

pit.show()
