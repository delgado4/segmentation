#!/usr/bin/python

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.layers import ConcatLayer
from lasagne.regularization import regularize_network_params, l2, l1
from lasagne.utils import floatX
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.misc
import scipy.ndimage
import sys
import time
import theano
import theano.tensor as T
import pickle
import platform
from neural_cnn import *

# For DEBUG
import pdb

PHOTO_WIDTH = 240
NUM_SLICES = 155
NUM_MODALITIES = 4
NUM_PATIENTS = 220

def get_cubes(volume, dim, num_pixels, slices, rows, cols):
	half_len = (dim-1)/2
	cubes = np.empty((num_pixels, dim, dim, dim))
	for i in range(0,slices.shape[0]):
		temp = volume[slices[i]-half_len:slices[i]+half_len+1,
						rows[i]-half_len:rows[i]+half_len+1,
						cols[i]-half_len:cols[i]+half_len+1]
		if(temp.shape != (33,33,33)):
			pdb.set_trace()
		cubes[i,:,:,:] = temp
	return cubes

def get_patient_volumes(folder, patient_num, T1_mean, T1c_mean, T2_mean, FLAIR_mean):
	if(platform.system() == 'Darwin'):
		T1c = folder + 'pat' + str(patient_num) + '/MR_T1c/'
	else:
		T1c = folder + 'pat' + str(patient_num) + '/MR_T1C/'
	T1c_volume = load_volume(T1c)
	T1c_volume -= T1c_mean

	# Load T1 images, X.shape = (num_pixels, dim, dim, dim)
	T1 = folder + 'pat' + str(patient_num) + '/MR_T1/'
	T1_volume = load_volume(T1) - T1_mean

	# Load T2 images, X.shape = (num_pixels, 3*dim, dim, dim)
	T2 = folder + 'pat' + str(patient_num) + '/MR_T2/'
	T2_volume = load_volume(T2) - T2_mean

	# Load FLAIR images, X.shape = (num_pixels, 4*dim, dim, dim)
	FLAIR = folder + 'pat' + str(patient_num) + '/MR_FLAIR/'
	FLAIR_volume = load_volume(FLAIR) - FLAIR_mean

	# Load OT labels, Y.shape = (num_pixels)
	OT = folder + 'pat' + str(patient_num) + '/OT/'
	OT_volume = load_volume(OT)

	return T1_volume, T1c_volume, T2_volume, FLAIR_volume, OT_volume

def init_classifier(network, num_classes = 2, input_var = None):
	output = lasagne.layers.get_output(network, deterministic=True)
	get_softmax_vals = theano.function([input_var], output)
	return get_softmax_vals

def load_network(filename, network):
	with np.load(filename) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)
	return network

def classify(softmax_pixels, targets, weights):
	predictions = np.argmax(softmax_pixels,axis=2)
	TP = np.sum((predictions==np.ones(targets.shape)).astype(np.int32) * (targets==np.ones(targets.shape)).astype(np.int32))
	FP = np.sum((predictions==np.ones(targets.shape)).astype(np.int32) * (targets==np.zeros(targets.shape)).astype(np.int32))
	TN = np.sum((predictions==np.zeros(targets.shape)).astype(np.int32) * (targets==np.zeros(targets.shape)).astype(np.int32))
	FN = np.sum((predictions==np.zeros(targets.shape)).astype(np.int32) * (targets==np.ones(targets.shape)).astype(np.int32))
	return TP,FP,TN,FN

def main(patient_num, num_classes = 2, slice_num = None):
	if(platform.system() == 'Darwin'):
		folder = '/Users/dominicdelgado/Documents/Radiogenomics/bratsHGG/jpeg/'
	else:
		#folder = '/home/ubuntu/Neural_CNN/jpeg/'
		folder = '/home/ubuntu/data/jpeg/'
	filename = 'cnn_params140.npz'

	# Set up Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

	# Create neural network
	network = build_cnn(input_var=input_var)

	# Load network parameters
	network = load_network(filename, network)

	# Create convnet function
	get_softmax_vals = init_classifier(network, num_classes, input_var)

	# Calculate mean images
	T1_mean, T1c_mean, T2_mean, FLAIR_mean = get_all_mean_volumes(folder)
	#T1_mean = np.zeros((NUM_SLICES,PHOTO_WIDTH,PHOTO_WIDTH))
	#T1c_mean = np.zeros((NUM_SLICES,PHOTO_WIDTH,PHOTO_WIDTH))
	#T2_mean = np.zeros((NUM_SLICES,PHOTO_WIDTH,PHOTO_WIDTH))
	#FLAIR_mean = np.zeros((NUM_SLICES,PHOTO_WIDTH,PHOTO_WIDTH))

	# Load mean-centered data
	T1, T1c, T2, FLAIR, OT = get_patient_volumes(folder, patient_num, T1_mean, T1c_mean, T2_mean, FLAIR_mean)

	slices = np.asarray(range(NUM_SLICES))
	if(slice_num != None):
		slices = np.asarray([slice_num])
	step = 15
	num_cols = PHOTO_WIDTH
	num_rows = step
	cols = np.arange(step*PHOTO_WIDTH) % PHOTO_WIDTH

	# Zero-pad volumes
	pad = 16
	T1c = np.pad(T1c, pad, 'constant')
	T1 = np.pad(T1, pad, 'constant')
	T2 = np.pad(T2, pad, 'constant')
	FLAIR = np.pad(FLAIR, pad, 'constant')
	cols += pad
	slices += pad
	num_slices = slices.shape[0]

	# Batch classification process
	print('Running examples through convnet...\n')
	softmax_pixels = np.zeros((num_slices,PHOTO_WIDTH,PHOTO_WIDTH,num_classes))
	index = 0
	for sl in slices:
		for start in range(0,PHOTO_WIDTH,step):
			# Pair each row with all columns
			rows = np.empty(0)
			for i in range(start, start+step):
				rows = np.concatenate((rows, i*np.ones(PHOTO_WIDTH)))

			# Account for zero-padding
			rows += pad

			#pdb.set_trace()

			T1c_cubes = get_cubes(T1c, 33, num_rows*num_cols, sl*np.ones(num_rows*num_cols), rows, cols)
			X = T1c_cubes

			T1_cubes = get_cubes(T1, 33, num_rows*num_cols, sl*np.ones(num_rows*num_cols), rows, cols)
			X = np.concatenate((X,T1_cubes),1)

			T2_cubes = get_cubes(T2, 33, num_rows*num_cols, sl*np.ones(num_rows*num_cols), rows, cols)
			X = np.concatenate((X,T2_cubes),1)

			FLAIR_cubes = get_cubes(FLAIR, 33, num_rows*num_cols, sl*np.ones(num_rows*num_cols), rows, cols)
			X = np.concatenate((X,FLAIR_cubes),1)

			X = X.astype(np.float32)
			results = get_softmax_vals(X)
			
			for i in range(len(rows)):
				softmax_pixels[index,rows[i]-pad,cols[i]-pad,:] = results[i,:]
		index+=1
	
	pdb.set_trace()
	'''
	# Classify for various weightings
	lambdas = 10 ** np.arange(-1,2,30)
	TP = np.zeros(len(lambdas))
	FP = np.zeros(len(lambdas))
	TN = np.zeros(len(lambdas))
	FN = np.zeros(len(lambdas))
	targets = (OT>0).astype(np.int32)

	print('Classifying...\n')
	for i in range(len(lambdas)):
		l = lambdas[i]
		weights = np.asarray([(1-l), l])
		tp, fp, tn, fn = classify(softmax_pixels, targets, weights)
		TP[i] += tp
		FP[i] += fp
		TN[i] += tn
		FN[i] += fn

	print('Done.\n')
	#pdb.set_trace()
	'''
	# Save results
	#data = np.concatenate((np.reshape(lambdas,(1,-1)), np.reshape(TP,(1,-1)), np.reshape(FP,(1,-1)), np.reshape(TN,(1,-1)), np.reshape(FN,(1,-1))),axis=0)
	#np.save('cnn_roc140_' + str(patient_num) + '_' + str(sl) + '_.npy',data)
	#np.save('softmax140' + str(patient_num) + '_' + str(sl) + '_.npy',softmax_pixels)
	np.save('softmax140_full' + str(patient_num)  + '_.npy',softmax_pixels)

if __name__ == '__main__':
	kwargs = {}
	if len(sys.argv) > 1:
		kwargs['patient_num'] = int(sys.argv[1])
	if len(sys.argv) > 2:
		kwargs['num_classes'] = int(sys.argv[2])
	if len(sys.argv) > 3:
		kwargs['slice_num'] = int(sys.argv[3])
	main(**kwargs)
