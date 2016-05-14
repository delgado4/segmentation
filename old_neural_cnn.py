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

'''
Given the data location, edge length 
of pixel neighborhood, and a set of patients, generates volumes from 
num_pixels random locations in each volume. The volumes are stacks of 
num_pixels x num_pixels x num_pixels cubes for each modality.
'''
def load_data(folder, patient_list, dim, num_pixels):
	def load_volume(folder):
		vol = np.zeros((NUM_SLICES, PHOTO_WIDTH, PHOTO_WIDTH))
		im = np.zeros((PHOTO_WIDTH, PHOTO_WIDTH))
		for slice in range(0,NUM_SLICES):
			im = scipy.misc.imread(folder + str(slice) + '.jpeg')
			vol[slice,:,:] = im
		return vol

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

	def sample_volume(folder, patient_num, dim, num_pixels):

		
		# Load T1c images, X.shape = (num_pixels, 2*dim, dim, dim)
		if(platform.system() == 'Darwin'):
			T1c = folder + 'pat' + str(patient_num) + '/MR_T1c/'
		else:
			T1c = folder + 'pat' + str(patient_num) + '/MR_T1C/'
		volume = load_volume(T1c)

		# Calculate locations of nonzero pixels in T1c
		half_len = (dim-1)/2
		nonzero_pixels = np.asarray([(sl,row,col) 
		for sl in range(NUM_SLICES) 
		for row in range(PHOTO_WIDTH) 
		for col in range(PHOTO_WIDTH) 
		if volume[sl,row,col] > 0
		if sl > half_len if sl < NUM_SLICES-half_len-1
		if row > half_len if row < PHOTO_WIDTH-half_len-1
		if col > half_len if col < PHOTO_WIDTH-half_len-1])

		# Sample pixels uniformly from nonzero pixels
		indices = np.random.choice(range(nonzero_pixels.shape[0]), num_pixels)
		slices = nonzero_pixels[indices,0]
		rows = nonzero_pixels[indices,1]
		cols = nonzero_pixels[indices,2]

		#pdb.set_trace()


		T1c_cubes = get_cubes(volume, dim, num_pixels, slices, rows, cols)
		X = T1c_cubes

		# Load T1 images, X.shape = (num_pixels, dim, dim, dim)
		T1 = folder + 'pat' + str(patient_num) + '/MR_T1/'
		volume = load_volume(T1)
		T1_cubes = get_cubes(volume, dim, num_pixels, slices, rows, cols)
		X = np.concatenate((X,T1_cubes),1)

		# Load T2 images, X.shape = (num_pixels, 3*dim, dim, dim)
		T2 = folder + 'pat' + str(patient_num) + '/MR_T2/'
		volume = load_volume(T2)
		T2_cubes = get_cubes(volume, dim, num_pixels, slices, rows, cols)
		X = np.concatenate((X,T2_cubes),1)

		# Load FLAIR images, X.shape = (num_pixels, 4*dim, dim, dim)
		FLAIR = folder + 'pat' + str(patient_num) + '/MR_FLAIR/'
		volume = load_volume(FLAIR)
		FLAIR_cubes = get_cubes(volume, dim, num_pixels, slices, rows, cols)
		X = np.concatenate((X,FLAIR_cubes),1)

		# Load OT labels, Y.shape = (num_pixels)
		OT = folder + 'pat' + str(patient_num) + '/OT/'
		volume = load_volume(OT)
		Y = np.empty(num_pixels)
		for i in range(num_pixels):
			Y[i] = volume[slices[i], rows[i], cols[i]]

		return X, Y

	# Build minibatch 4-D data array (examples, edge length*4, length, length)
	# and 1-D observation array
	num_patients = patient_list.shape[0]
	X = np.empty((num_pixels*num_patients, dim*NUM_MODALITIES, dim, dim))
	Y = np.empty(num_pixels*num_patients)

	index = 0
	for patient in patient_list:
		X_samples, Y_samples = sample_volume(folder, patient, dim, num_pixels)
		X[index:index+num_pixels,:,:,:] = X_samples
		Y[index:index+num_pixels] = Y_samples
		index += 1
	
	# We only care about whether this is cancer or not right now
	Y = Y>0
	return X, Y.astype(np.int32)

def build_cnn(filter_size = 33, num_neurons = 128, num_classes = 5, 
				input_var = None):
	network = InputLayer(
			shape=(None, NUM_MODALITIES*filter_size, filter_size, filter_size),
			input_var=input_var)
	network = ConvLayer(network, num_filters=72, filter_size=filter_size)
	network = NormLayer(network)
	network = DenseLayer(DropoutLayer(network, p=0.5), num_units=num_neurons)
	network = DenseLayer(DropoutLayer(network, p=0.5), num_units=num_neurons)
	network = DenseLayer(DropoutLayer(network, p=0.5), num_units=num_classes,
							nonlinearity=lasagne.nonlinearities.softmax)
	return network

def train_net(folder, train_set, validation_set, test_set, edge_len, num_epochs = 500, 
				l1_reg = 0, l2_reg = 0):
	'''
	Note: The following setup code is from the mnist.py code from lasagne examples.
	The testing structure was changed to match my data setup
	'''
	
	# Set up Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

	# Create neural network
	network = build_cnn(input_var=input_var)

	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var) + l1_reg*lasagne.regularization.regularize_network_params(network, l1) + l2_reg*lasagne.regularization.regularize_network_params(network, l2)
	loss = loss.mean()
	# We could add some weight decay as well here, see lasagne.regularization.

	# Perform updates using Adam
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.adam(loss, params, learning_rate=0.001)

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

	patients_per_batch = 1
	pixels_per_batch = 3300
	pixels_per_patient = pixels_per_batch*1 
	iterations_per_patient = int(np.ceil(pixels_per_patient/(pixels_per_batch*patients_per_batch)))


	print("Starting training...")
	for epoch in range(num_epochs):
		'''
		Temporarily reduce # patients/iteration to 1
		'''
		train_index = map(int,np.floor(np.random.rand(1)*np.asarray(train_set).shape[0]))
		train_batch = [train_set[i] for i in train_index]

		val_index = map(int,np.floor(np.random.rand(1)*np.asarray(train_set).shape[0]))
		val_batch = [train_set[i] for i in train_index]

		print("Starting epoch " + str(epoch))
		train_err = 0
		train_acc = 0
		train_batches = 0
		start_time = time.time()
		# "Full pass" over patients
		for patient in train_batch:
			for i in range(iterations_per_patient):
				inputs, targets = load_data(folder, np.asarray([patient]), edge_len, pixels_per_batch)
				#pdb.set_trace()
				assert len(inputs) == len(targets)
				print 'Training on patient %d...' % patient
				train_err += train_fn(inputs, targets)
				print 'Computing training accuracy'
				train_acc += train_acc_fn(inputs, targets)
				train_batches += 1

        	# And a "full pass" over the validation data:
        	val_err = 0
        	val_acc = 0
        	val_batches = 0
		print 'Validating...'
        	for patient in val_batch:
        		for i in range(iterations_per_patient):
	            		inputs, targets = load_data(folder, np.asarray([patient]), edge_len, pixels_per_batch)
	            		err, acc = val_fn(inputs, targets)
	            		val_err += err
	            		val_acc += acc
	            		val_batches += 1

        	# Then we print the results for this epoch:
        	print("Epoch {} of {} took {:.3f}s".format(
            		epoch + 1, num_epochs, time.time() - start_time))
        	print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        	print("  training accuracy:\t\t{:.2f} %".format(
            		train_acc / val_batches * 100))
        	print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        	print("  validation accuracy:\t\t{:.2f} %".format(
            		val_acc / val_batches * 100))

    	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	print 'Testing...'
	for patient in test_set:
		for i in range(iterations_per_patient):
			inputs, targets = load_data(folder, np.asarray([patient]), edge_len, pixels_per_batch)
			err, acc = val_fn(inputs, targets)
			test_err += err
			test_acc += acc
			test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("  test accuracy:\t\t{:.2f} %".format(
		test_acc / test_batches * 100))

	return val_acc/val_batches, test_acc/test_batches

def main(num_epochs=10,percent_validation=0.05,percent_test=0.10,edge_len=33,
			num_regularization_params = 20):
	rng_state = np.random.get_state()

	if(platform.system() == 'Darwin'):
		folder = '/Users/dominicdelgado/Documents/Radiogenomics/bratsHGG/jpeg/'
	else:
		folder = '/home/ubuntu/data/jpeg/'

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
	#l1_reg = 10**(np.random.rand(num_regularization_params)*4 - 4)
	#l2_reg = 10**(np.random.rand(num_regularization_params)*4 - 4)
	l1_reg = np.asarray([0])
	l2_reg = np.asarray([0])

	best_l1 = l1_reg[0]
	best_l2 = l2_reg[0]
	best_test_pct = 0
	best_val_pct = 0
	data_valid = False

	# Train network
	for i in range(l2_reg.shape[0]):
		val_pct, test_pct = train_net(folder = folder, train_set=train_set, 
					validation_set=validation_set, test_set=test_set, 
					num_epochs = num_epochs, l1_reg = l1_reg, l2_reg = l2_reg,
					edge_len = edge_len)
		if (not data_valid) or (test_pct > best_test_pct):
			best_l1 = l1_reg[0]
			best_l2 = l2_reg[0]
			best_test_pct = 0
			best_val_pct = 0
			data_valid = True

	# Report results and save
	print "Achieved test error of %f with l1 = %f and l2 = %f." % (best_test_pct, best_l1, best_l2)

	with open(folder + 'rng_state.dat') as f:
		pickle(rng_state, f)

	return 0

if __name__ == '__main__':
	kwargs = {}
	if len(sys.argv) > 1:
		kwargs['num_epochs'] = int(sys.argv[1])
	main(**kwargs)
