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

# For DEBUG
import pdb

PHOTO_WIDTH = 240
NUM_SLICES = 155
NUM_MODALITIES = 4
NUM_PATIENTS = 220
UNIMPLEMENTED = None

def BinaryImageCrossEntropyLoss():
	return UNIMPLEMENTED


def load_data(folder, patient_list, dim, num_pixels):
	def load_volume(folder):
		vol = np.zeros((NUM_SLICES, PHOTO_WIDTH, PHOTO_WIDTH))
		im = np.zeros((PHOTO_WIDTH, PHOTO_WIDTH))
		for slice in range(0,NUM_SLICES):
			im = scipy.misc.imread(folder + str(slice) + '.jpeg')
			vol[slice,:,:] = im
		return vol

	# Load T1c images, X.shape = 
	T1c = folder + 'pat' + str(patient_num) + '/MR_T1c/'
	T1c_volume = np.reshape(load_volume(T1c), (-1, 1, PHOTO_WIDTH, PHOTO_WIDTH))

	# Load T1 images, X.shape = 
	T1 = folder + 'pat' + str(patient_num) + '/MR_T1/'
	T1_volume = np.reshape(load_volume(T1), (-1, 1, PHOTO_WIDTH, PHOTO_WIDTH))

	# Load T2 images, X.shape = 
	T2 = folder + 'pat' + str(patient_num) + '/MR_T2/'
	T2_volume = np.reshape(load_volume(T2), (-1, 1, PHOTO_WIDTH, PHOTO_WIDTH))

	# Load FLAIR images, X.shape = 
	FLAIR = folder + 'pat' + str(patient_num) + '/MR_FLAIR/'
	FLAIR_volume = np.reshape(load_volume(FLAIR), (-1, 1, PHOTO_WIDTH, PHOTO_WIDTH))

	# Load OT labels, Y.shape = 
	OT = folder + 'pat' + str(patient_num) + '/OT/'
	OT_volume = np.reshape(load_volume(OT), (-1, 1, PHOTO_WIDTH, PHOTO_WIDTH))


def build_net():
	return UNIMPLEMENTED

def train_net():
	return UNIMPLEMENTED

def main(num_epochs=500):
	return 0

if __name__ == '__main__':
	kwargs = {}
	if len(sys.argv) > 1:
		kwargs['num_epochs'] = int(sys.argv[1])
	main(**kwargs)