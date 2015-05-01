# This is all 'experimental code'; this will generate SVM input data for all the positive examples
# ie. for all the logo examples from the 'matlab/training_data/' folders

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


# Make sure that caffe is on the python path:
caffe_root = '../../../caffe-master/'
pckl_path = '../../PCKL/'

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

#generate positive svm training input
print('generating and loading positive svm training inputs...')
training_root = '../matlab/training_data/'

logos = os.listdir(training_root)
logos.remove('.DS_Store')

distortion_size = 100

X_pos = np.zeros((len(logos)*distortion_size,4096))
Y_pos = np.zeros(len(logos)*distortion_size)


k_index = 0;
for i in range(len(logos)):

	print(str(i*100.0/len(logos)) + '%')

	logo_type = logos[i]
	logo_type_distortions = os.listdir(training_root + logo_type + '/')

	for j in range(len(logo_type_distortions)):

#		print(j)
		input_image = caffe.io.load_image(training_root + logo_type + '/' + logo_type_distortions[j]);

		net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
		out = net.forward()
		feat = net.blobs['fc7'].data[0]
	
		X_pos[k_index,:] = feat
		Y_pos[k_index] = i + 1
		
		k_index = k_index + 1


#import pickle

f = open('X_pos.pckl', 'w')
pickle.dump(X_pos, f)
f.close()

f = open('Y_pos.pckl', 'w')
pickle.dump(Y_pos, f)
f.close()


#generate negative svm training input
print('generating and loading negative svm training...')
training_root = '../ImageNet_10000/'
image_net = os.listdir(training_root)
image_net.remove('.DS_Store')

X_neg = np.zeros((len(image_net),4096))
Y_neg = np.zeros(len(image_net))

for i in range(len(image_net)):

	if (i%100 == 0):
		print(str(i*100.0/len(image_net)) + '%')
	
	input_image = caffe.io.load_image(training_root + image_net[i]);

	net.blobs['data'].data[...] = transformer.preprocess('data', input_image)
	out = net.forward()
	feat = net.blobs['fc7'].data[0]
	
	X_neg[i,:] = feat
	Y_neg[i] = 0

f = open(pckl_path + 'X_neg.pckl', 'w')
pickle.dump(X_neg, f)
f.close()

f = open(pckl_path + 'Y_neg.pckl', 'w')
pickle.dump(Y_neg, f)
f.close()

print(np.size(X_neg))
print(np.size(Y_neg))


#print(len(logos))

#print(logos)


print '... hello world'
