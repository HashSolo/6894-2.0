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

import math

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

phish_zoo_dir = '../Phish/Key Cases/'

phish_zoo = os.listdir(phish_zoo_dir)

if '.DS_Store' in phish_zoo:
	phish_zoo.remove('.DS_Store')

print(phish_zoo)

for i in range(0,len(phish_zoo)):

	phish_file = phish_zoo[i]

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
	print('File	' + str(i) +':		' + phish_file) #db
	
	#read in image
	contains_logo = 0
	input_image = caffe.io.load_image(phish_zoo_dir + phish_file);
	
	h = len(input_image[:,1,1])
	w = len(input_image[1,:,1])

	#HACKY sliding window

	scales = [.7]
	shapes = [.30]

	#used for day long run
#	scales = [.33, .5, .66, .7, 1.0]; 
#	shapes = [.30, .34, .45, .56, .75, .8, 1.0];
	
	#scales = [.2, .33, .5, .6, .8];
	#shapes = [.3, .4,.45, .55, 1.0] # ratio of height to width

	x_stride = 1
	y_stride = 1


	#yes it's four nested loops
	for scales_i in range(len(scales)):

		print('		Scale: ' + str(scales[scales_i]))
		im_h = math.trunc(scales[scales_i] * h)

		for shapes_i in range(len(shapes)):


			print('			Shape: ' + str(shapes[shapes_i]))
			im_w = math.trunc(im_h / shapes[shapes_i])

			if (im_w > w):
				continue

			for y_i in range(math.trunc((h-im_h)*0.40),
							 math.trunc((h - im_h)*0.60)):

				if (y_i % y_stride != 0):
					continue

				#if (y_i + im_h > h)
				print('				' + 
					str(math.trunc(y_i*100.0 / (h - im_h))) + '%')

				for x_i in range(math.trunc((w-im_w)*0.0),
				 				 math.trunc((w - im_w)*0.1)):

					if (x_i % x_stride != 0):
						continue

					sub_im = input_image[y_i:y_i+im_h, 
										 x_i:x_i+im_w, :]

					#plt.imshow(sub_im) #db
					#plt.show() #dbg

					#run feed-forward CNN
					net.blobs['data'].data[...] = transformer.preprocess('data', sub_im)
					out = net.forward()
					feat = net.blobs['fc7'].data[0]

					prediction = svm_a.predict(feat)
					if (prediction != 0):
						print('x: ' + str(x_i) +
							  ', y: ' + str(y_i))
						print('PREDICTION: '+ str(prediction))
						contains_logo = 1

	print(contains_logo)


			#		print('x_i' + str(x_i))
			#		print('y_i' + str(y_i))



		
	

	
	#predict class with svm
#	print('			prediction: ' + str(prediction))
	
#	plt.imshow(input_image) #db
#	plt.show() #dbg
		
		
		

print '... hello world'
