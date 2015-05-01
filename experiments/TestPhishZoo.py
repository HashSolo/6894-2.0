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

print('Setting up Caffe and imagenet network...')
# Make sure that caffe is on the python path:
caffe_root = '../../../caffe-master/'

import sys
sys.path.insert(0, caffe_root + 'python')


import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    os.system(caffe_root + 'scripts/download_model_binary.py ' + caffe_root + 'models/bvlc_reference_caffenet')

caffe.set_mode_cpu()


net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
	                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
	                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

net.blobs['data'].reshape(1,3,227,227)

#load svms
print('loading svms...')
f = open('svm4700_C0_Gneg4.pckl')
svm_a = pickle.load(f)
f.close()

f = open('svm4700_C10_Gneg9.pckl')
svm_b = pickle.load(f)
f.close()

print('classifying data...')

phish_zoo_dir = '../Phish/Logo/'

phish_zoo = os.listdir(phish_zoo_dir)

if '.DS_Store' in phish_zoo:
	phish_zoo.remove('.DS_Store')

print(phish_zoo)

for i in range(562,len(phish_zoo)):

	print('Folder: ' + str(i) + ' -		' + phish_zoo[i]) #db

	phish_folder = phish_zoo[i]
	phish_subfolders = os.listdir(phish_zoo_dir + phish_folder + '/')
	
	for j in range(len(phish_subfolders)):
		
		phish_file = phish_subfolders[j]

#		print(str(j)) #db
#		print(phish_file)  #db

		#filter out files without file-types
		phish_split = phish_file.split(".")
		if len(phish_split) == 1:
			continue
		
		#filter out non-image fiels
		file_type = phish_split[1]
		if ((file_type != 'jpg') & (file_type != 'gif') & 
		    (file_type != 'png') & (file_type != 'ico')):
			continue
			
		im_name = phish_file[0]
		print('		' + phish_file) #db
		
		#read in image
		input_image = caffe.io.load_image(phish_zoo_dir + 
		phish_folder + '/' + phish_file);
		
		#run feed-forward CNN
		net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
		out = net.forward()
		feat = net.blobs['fc7'].data[0]
		
		#predict class with svm
		prediction = svm_a.predict(feat)
		print('			prediction: ' + str(prediction))
		
#		plt.imshow(input_image) #db
#		plt.show() #dbg
		
		
		

print '... hello world'
