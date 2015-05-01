import cPickle
import pickle
import gzip
import os
import sys
import time

import scipy
import scipy.ndimage

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

pckl_path = '../../PCKL/'

print('loading X_4700 svm training...') 
f = open(pckl_path + 'X_4700.pckl')
X_4700 = pickle.load(f)
f.close()

print('loading Y_4700 svm training labels...') 
f = open(pckl_path + 'Y_4700.pckl')
Y_4700 = pickle.load(f)
f.close()

print('training svm4700_C10_Gneg9 ...') 
clf = svm.SVC(C = 10**10, gamma = 10**-9)
print(clf.fit(X_4700, Y_4700))

print('saving svm4700_C10_Gneg9 ...') 
f = open(pckl_path + 'svm4700_C10_Gneg9.pckl', 'w')
pickle.dump(clf, f)
f.close()

print('training svm4700_C0_Gneg4 ...') 
clf = svm.SVC(C = 10**0, gamma = 10**-4)
print(clf.fit(X_4700, Y_4700))

print('saving svm4700_C0_Gneg4 ...') 
f = open(pckl_path + 'svm4700_C0_Gneg4.pckl', 'w')
pickle.dump(clf, f)
f.close()


