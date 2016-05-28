import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import batch_norm
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

# For DEBUG
import pdb

PHOTO_WIDTH = 240
NUM_SLICES = 155
NUM_MODALITIES = 4
NUM_PATIENTS = 220

def get_mean_volume(folder, modality):
	mean_volume = np.zeros((NUM_SLICES, PHOTO_WIDTH, PHOTO_WIDTH))
	patient_volume = np.zeros((NUM_SLICES, PHOTO_WIDTH, PHOTO_WIDTH))
	for patient_num in range(NUM_PATIENTS):
		path = folder + 'pat' + str(patient_num) + '/' + modality + '/'
		patient_volume = load_volume(path)
		mean_volume += patient_volume.astype(np.float32) / NUM_PATIENTS
	return mean_volume

def get_all_mean_volumes(folder):
	print('Calculating mean images...')
	print('MR_T1')
	T1_mean = get_mean_volume(folder, 'MR_T1')
	print('MR_T1C')
	if(platform.system() == 'Darwin'):
		T1c_mean = get_mean_volume(folder, 'MR_T1c')
	else:
		T1c_mean = get_mean_volume(folder, 'MR_T1C')
	print('MR_T2')
	T2_mean = get_mean_volume(folder, 'MR_T2')
	print('MR_FLAIR')
	FLAIR_mean = get_mean_volume(folder, 'MR_FLAIR')
	return T1_mean, T1c_mean, T2_mean, FLAIR_mean

def load_volume(folder):
	vol = np.zeros((NUM_SLICES, PHOTO_WIDTH, PHOTO_WIDTH))
	im = np.zeros((PHOTO_WIDTH, PHOTO_WIDTH))
	for slice in range(0,NUM_SLICES):
		im = scipy.misc.imread(folder + str(slice) + '.jpeg')
		vol[slice,:,:] = im	
	return vol

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

def get_sample_indices(nonzero_pixels, num_pixels):
	# Sample pixels uniformly from nonzero pixels
	indices = np.random.choice(range(nonzero_pixels.shape[0]), num_pixels)
	slices = nonzero_pixels[indices,0]
	rows = nonzero_pixels[indices,1]
	cols = nonzero_pixels[indices,2]
	return slices, rows, cols

def get_nonzero_pixels(volume, dim):
	half_len = (dim-1)/2
	nonzero_pixels = np.asarray([(sl,row,col) 
	for sl in range(NUM_SLICES) 
	for row in range(PHOTO_WIDTH) 
	for col in range(PHOTO_WIDTH) 
	if volume[sl,row,col] > 0
	if sl > half_len if sl < NUM_SLICES-half_len-1
	if row > half_len if row < PHOTO_WIDTH-half_len-1
	if col > half_len if col < PHOTO_WIDTH-half_len-1])
	return nonzero_pixels

def get_balanced_batch(T1c_nonzero, OT_nonzero, dim, num_pixels,T1, T1c, T2, FLAIR, OT):
	assert(num_pixels%2 == 0)

	# Get 50% indices from nonzero T1c (mostly non-cancer)
	slices_T1c, rows_T1c, cols_T1c = get_sample_indices(T1c_nonzero, num_pixels/2)

	# Get 50% indices from nonzero OT (cancer)
	slices_OT, rows_OT, cols_OT = get_sample_indices(OT_nonzero, num_pixels/2)

	slices = np.concatenate((slices_T1c, slices_OT))
	rows = np.concatenate((rows_T1c, rows_OT))
	cols = np.concatenate((cols_T1c, cols_OT))

	T1c_cubes = get_cubes(T1c, dim, num_pixels, slices, rows, cols)
	X = T1c_cubes

	T1_cubes = get_cubes(T1, dim, num_pixels, slices, rows, cols)
	X = np.concatenate((X,T1_cubes),1)

	T2_cubes = get_cubes(T2, dim, num_pixels, slices, rows, cols)
	X = np.concatenate((X,T2_cubes),1)

	FLAIR_cubes = get_cubes(FLAIR, dim, num_pixels, slices, rows, cols)
	X = np.concatenate((X,FLAIR_cubes),1)

	Y = np.empty(num_pixels)
	for i in range(num_pixels):
		Y[i] = OT[slices[i], rows[i], cols[i]]

	Y = Y>0
	return X.astype(np.float32), Y.astype(np.int32)

# Note that dim must be an odd number
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

def build_cnn(filter_size = 33, num_neurons = 1024, num_classes = 2, 
				input_var = None):
	network = InputLayer(
			shape=(None, NUM_MODALITIES*filter_size, filter_size, filter_size),
			input_var=input_var)
	network = ConvLayer(network, num_filters=72, filter_size=filter_size)
	network = NormLayer(network)
	network = (DenseLayer(network, num_units=num_neurons))
	network = (DenseLayer(DropoutLayer(network, p=0.5), num_units=num_neurons))
	network = DenseLayer(DropoutLayer(network, p=0.5), num_units=num_classes,
							nonlinearity=lasagne.nonlinearities.softmax)
	return network

def save_data(filename, data):
	output_file = open(filename, 'w')
	output_file.write(','.join(map(str, data)))
	output_file.close()

def train_net(folder, train_set, validation_set, test_set, edge_len, num_epochs = 500, 
				l1_reg = 0, l2_reg = 0, learn_rate = 0.001, 
				T1_mean = None, T1c_mean = None, T2_mean = None, FLAIR_mean = None):
	'''
	Note: The following setup code is from the mnist.py code from lasagne examples.
	The testing structure was changed to match my data setup
	'''
	
	# Set up Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

	# Create neural network
	network = build_cnn(input_var=input_var)

	# Load params from file
	# with np.load('cnn_params_large140.npz') as f:
	# 	param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	# lasagne.layers.set_all_param_values(network, param_values)

	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var) + l1_reg*lasagne.regularization.regularize_network_params(network, l1) + l2_reg*lasagne.regularization.regularize_network_params(network, l2)
	loss = loss.mean()
	# We could add some weight decay as well here, see lasagne.regularization.

	# Perform updates using Adam
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.adam(loss, params, learning_rate=learn_rate)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers. 
	# Here we're using so-called "elastic net" regulization
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
															target_var) + l1_reg*lasagne.regularization.regularize_network_params(network, l1) + l2_reg*lasagne.regularization.regularize_network_params(network, l2)
	test_loss = test_loss.mean()
	# As a bonus, also create an expression for the classification accuracy:
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
						dtype=theano.config.floatX)

	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	train_acc_fn = theano.function([input_var, target_var], test_acc)

	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	patients_per_batch = 2
	pixels_per_batch = 500
	iterations_per_patient = 10
	pixels_per_patient = pixels_per_batch*iterations_per_patient
	
	#int(np.ceil(pixels_per_patient/(pixels_per_batch*patients_per_batch)))
	val_patients_per_batch = 1

	val_loss_hist = []
	train_loss_hist = []
	val_acc_hist = []
	train_acc_hist = []

	print("Starting training...")
	for epoch in range(num_epochs):
		train_index = map(int,np.floor(np.random.rand(patients_per_batch)*np.asarray(train_set).shape[0]))
		train_batch = [train_set[i] for i in train_index]

		val_index = map(int,np.floor(np.random.rand(val_patients_per_batch)*np.asarray(validation_set).shape[0]))
		val_batch = [validation_set[i] for i in val_index]

		print("Starting epoch " + str(epoch+1))
		train_err = 0
		train_acc = 0
		train_batches = 0
		start_time = time.time()
		# "Full pass" over patients
		for patient in train_batch:
			# Load mean-centered data
			T1, T1c, T2, FLAIR, OT = get_patient_volumes(folder, patient, T1_mean, T1c_mean, T2_mean, FLAIR_mean)
			T1c_nonzero = get_nonzero_pixels(T1c, edge_len)
			OT_nonzero = get_nonzero_pixels(OT, edge_len)
			
			print 'Training on patient %d...' % patient
			for i in range(iterations_per_patient):
				inputs, targets = get_balanced_batch(T1c_nonzero, OT_nonzero, edge_len, pixels_per_batch,T1, T1c, T2, FLAIR, OT)
				assert len(inputs) == len(targets)
				err = train_fn(inputs, targets)
				acc = train_acc_fn(inputs, targets)
				train_err += err
				train_acc += acc
				train_batches += 1
				
				if(err >= 1 or acc >= 1):
					pdb.set_trace

		# And a "full pass" over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		for patient in val_batch:
			print 'Validating on patient %d...' % patient
			T1, T1c, T2, FLAIR, OT = get_patient_volumes(folder, patient, T1_mean, T1c_mean, T2_mean, FLAIR_mean)
			T1c_nonzero = get_nonzero_pixels(T1c, edge_len)
			OT_nonzero = get_nonzero_pixels(OT, edge_len)
			for i in range(iterations_per_patient):
				inputs, targets = get_balanced_batch(T1c_nonzero, OT_nonzero, edge_len, pixels_per_batch,T1, T1c, T2, FLAIR, OT)
				err, acc = val_fn(inputs, targets)
				val_err += err
				val_acc += acc
				val_batches += 1
				
				if(err >= 1 or acc >= 1):
					pdb.set_trace

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		print("  training accuracy:\t\t{:.2f} %".format(
			train_acc / train_batches * 100))
		print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("  validation accuracy:\t\t{:.2f} %".format(
			val_acc / val_batches * 100))
	
		# Save intermediate results every now and then
		train_loss_hist.append(train_err/train_batches)
		val_loss_hist.append(val_err/val_batches)
		train_acc_hist.append(train_acc/train_batches)
		val_acc_hist.append(val_acc/val_batches)

		if((epoch+1) % 10 == 0):
			save_data('train_loss.dat', train_loss_hist)
			save_data('val_loss.dat', val_loss_hist)
			save_data('train_acc.dat', train_acc_hist)
			save_data('val_acc.dat', val_acc_hist)
			
		if((epoch+1) % 25 == 0):
			np.savez('cnn_params_large' + str(epoch+1) + '.npz', *lasagne.layers.get_all_param_values(network))

    # After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	print 'Testing...'
	for patient in test_set:
		T1, T1c, T2, FLAIR, OT = get_patient_volumes(folder, patient, T1_mean, T1c_mean, T2_mean, FLAIR_mean)
		T1c_nonzero = get_nonzero_pixels(T1c, edge_len)
		OT_nonzero = get_nonzero_pixels(OT, edge_len)
		for i in range(iterations_per_patient):
			inputs, targets = get_balanced_batch(T1c_nonzero, OT_nonzero, edge_len, pixels_per_batch,T1, T1c, T2, FLAIR, OT)
			err, acc = val_fn(inputs, targets)
			test_err += err
			test_acc += acc
			test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("  test accuracy:\t\t{:.2f} %".format(
		test_acc / test_batches * 100))

	return val_acc/val_batches, test_acc/test_batches, lasagne.layers.get_all_param_values(network)

def main(num_epochs=500,percent_validation=0.05,percent_test=0.10,edge_len=33,
			num_regularization_params = 10):
	rng_state = np.random.get_state()

	if(platform.system() == 'Darwin'):
		folder = '/Users/dominicdelgado/Documents/Radiogenomics/bratsHGG/jpeg/'
	else:
		folder = '/home/ubuntu/data/jpeg/'

	# Calculate mean images
	T1_mean, T1c_mean, T2_mean, FLAIR_mean = get_all_mean_volumes(folder)
	# T1_mean = np.zeros((NUM_SLICES,PHOTO_WIDTH,PHOTO_WIDTH))
	# T1c_mean = np.zeros((NUM_SLICES,PHOTO_WIDTH,PHOTO_WIDTH))
	# T2_mean = np.zeros((NUM_SLICES,PHOTO_WIDTH,PHOTO_WIDTH))
	# FLAIR_mean = np.zeros((NUM_SLICES,PHOTO_WIDTH,PHOTO_WIDTH))

	# Generate test, training, and validation sets
	patient_list = range(NUM_PATIENTS)
	np.random.shuffle(patient_list)
	num_validation = int(np.floor(NUM_PATIENTS*percent_validation))
	num_test = int(np.floor(NUM_PATIENTS*percent_test))
	num_train = NUM_PATIENTS - num_test - num_validation

	train_set = patient_list[:num_train]
	test_set = patient_list[num_train:num_train+num_test]
	validation_set = patient_list[num_train+num_test:]

	# Try some different parameters from the range 1e-6 to 1e-2
	#l1_reg = 10**(-1*(np.random.rand(num_regularization_params)*4 + 3))
	#l2_reg = 10**(-1*(np.random.rand(num_regularization_params)*4 + 3))
	#lr = 10 ** (-1 * (np.random.rand(num_regularization_params) * 3 + 3.5))
	#lr = lr.astype(np.float32)
	#l1_reg = np.asarray([0.000070])
	#l2_reg = np.asarray([0.000025])
	#lr = np.asarray([0.0001])
	#lr = lr.astype(np.float32)
	l1_reg = np.asarray([0.0])
	l2_reg = np.asarray([0.0])
	lr = np.asarray([0.001]) #0.001])
	lr = lr.astype(np.float32)

	best_l1 = l1_reg[0]
	best_l2 = l2_reg[0]
	best_test_pct = 0
	best_val_pct = 0
	best_lr = 0
	data_valid = False
	best_params = None

	# Train network
	for i in range(l2_reg.shape[0]):
		val_pct, test_pct, params = train_net(folder = folder, train_set=train_set, 
					validation_set=validation_set, test_set=test_set, 
					num_epochs = num_epochs, l1_reg = l1_reg[i], 
					l2_reg = l2_reg[i], learn_rate = lr[i],
					edge_len = edge_len,
					T1_mean=T1_mean, T1c_mean=T1c_mean, T2_mean=T2_mean, 
					FLAIR_mean=FLAIR_mean)
		if (not data_valid) or (test_pct > best_test_pct):
			best_l1 = l1_reg[i]
			best_l2 = l2_reg[i]
			best_l4 = lr[i]
			best_test_pct = test_pct
			best_val_pct = val_pct
			best_params = params
			data_valid = True

		# Report results and save
		print "Achieved test error of %f with l1 = %f, l2 = %f, learn rate = %f." % (test_pct, l1_reg[i], l2_reg[i], lr[i])
		print "Best so far: %f with l1 = %f, l2 = %f, learn rate = %f." % (best_test_pct, best_l1, best_l2, best_lr)

	np.savez('cnn_params.npz', *best_params)
	return 0

if __name__ == '__main__':
	kwargs = {}
	if len(sys.argv) > 1:
		kwargs['num_epochs'] = int(sys.argv[1])
	main(**kwargs)
