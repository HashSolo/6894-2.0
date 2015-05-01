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

#generate svm training for X_4700 case

print('loading positive svm training...')

f = open(pckl_path + 'X_pos.pckl')
X_pos = pickle.load(f)
f.close()

print('assert correct size of positive data')
print(np.shape(X_pos))

print('loading negative svm training (may take a minute)...')
f = open(pckl_path + 'X_neg.pckl')
X_neg = pickle.load(f)
f.close()

print('assert correct size of negative data')
print(np.shape(X_neg))

sub_neg = X_neg[0:2000, :]

print('assert correct size of negative subset')
print(np.shape(sub_neg))

print('combining positive and negative svm training')
X_4700 = np.concatenate((X_pos, sub_neg))

print('assert correct size of X_4700')
print(np.shape(X_4700))

print('saving X_4700...')
f = open(pckl_path + 'X_4700.pckl', 'w')
pickle.dump(X_4700, f)
f.close()

print('loading positive svm training label...')
f = open(pckl_path + 'Y_pos.pckl')
Y_pos = pickle.load(f)
f.close()
print(np.shape(Y_pos))


print('extending labels with negatives')
ext = np.zeros(2000)
Y_4700 = np.concatenate((Y_pos,ext),axis=1)

print('verify the shape of the Y_4700 array; should be (4700)')
print(np.shape(Y_4700))

print('saving Y_4700...')
f = open(pckl_path + 'Y_4700.pckl', 'w')
pickle.dump(Y_4700, f)
f.close()


