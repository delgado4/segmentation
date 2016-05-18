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
UNIMPLEMENTED = None

'''
Take in k x n x n matrix of softmax predictions and 1-hot k x n x n target matrix
and compute the categorical cross-entropy sum(targets .* log(prediction))
'''

def ImageSoftmax(prediction,num_classes):
	'''
	prediction: 4-D tensor with dimensions (batch_size, num_classes, width, height)
	'''
	return T.exp(prediction) / T.tile(T.sum(T.exp(prediction), axis=1), [1,num_classes,1,1])

def ImageCrossEntropyLoss(prediction,targets,num_classes):
	'''
	prediction: 4-D tensor post-softmax
	targets: 3-D tensor of ground truth images
	'''
	loss_image = -T.eq(targets,0) * T.log(prediction[:,0,:,:])
	for i in range(1,num_classes):
		loss_image += -T.eq(targets,i) * T.log(prediction[:,i,:,:])
	return lasagne.objectives.aggregate(loss_image)


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

def get_slice_neighborhood(folder, slice, dim, mean_volume):
	vol = np.zeros((dim, PHOTO_WIDTH, PHOTO_WIDTH))
	im = np.zeros((PHOTO_WIDTH, PHOTO_WIDTH))
	half_len = (dim-1)/2
	index = 0
	for sl in range(slice-half_len,slice+half_len+1):
			im = scipy.misc.imread(folder + str(sl) + '.jpeg')
			vol[index,:,:] = im - mean_volume[sl,:,:]
			index += 1

def load_examples(folder, slices, dim, mean_volume):
		num_slices = slices.shape[0]
		vol = np.zeros((num_slices, dim, PHOTO_WIDTH, PHOTO_WIDTH))
		im = np.zeros((dim, PHOTO_WIDTH, PHOTO_WIDTH))
		index = 0
		for slice in slices:
			im = get_slice_neighborhood(folder,slice,dim,mean_volume)
			vol[index,:,:,:] = im
			index += 1
		return vol

def load_volume(folder):
		vol = np.zeros((NUM_SLICES, PHOTO_WIDTH, PHOTO_WIDTH))
		im = np.zeros((PHOTO_WIDTH, PHOTO_WIDTH))
		for slice in range(0,NUM_SLICES):
			im = scipy.misc.imread(folder + str(slice) + '.jpeg')
			vol[slice,:,:] = im
		return vol


def load_data(folder, patient_list, dim, num_slices, 
				T1_mean, T1c_mean, T2_mean, FLAIR_mean):
	# Initialize data tensors with empty tensors
	X = np.empty((0, dim * NUM_MODALITIES, PHOTO_WIDTH, PHOTO_WIDTH))
	Y = np.empty((0,PHOTO_WIDTH,PHOTO_WIDTH))
	min_slice = (dim-1)/2 + 1


	for patient_num	in patient_list:
		# Select slices uniformly at random from patient volume
		slices = np.floor(np.random.rand(num_slices)*(NUM_SLICES-dim) + min_slice).astype(np.int32)
		
		# Load T1c images, X.shape =  (num_slices,dim,PHOTO_WIDTH,PHOTO_WIDTH)
		if(platform.system() == 'Darwin'):
			T1c = folder + 'pat' + str(patient_num) + '/MR_T1c/'
		else:
			T1c = folder + 'pat' + str(patient_num) + '/MR_T1C/'
		T1c_volume = load_examples(folder, slices, dim, T1c_mean)
		patient_X = T1c_volume

		# Load T1 images, X.shape =  (num_slices,2dim,PHOTO_WIDTH,PHOTO_WIDTH)
		T1 = folder + 'pat' + str(patient_num) + '/MR_T1/'
		T1_volume = load_examples(folder, slices, dim, T1_mean)
		patient_X = np.concatenate((patient_X, T1_volume),1)

		# Load T2 images, X.shape =  (num_slices,3dim,PHOTO_WIDTH,PHOTO_WIDTH)
		T2 = folder + 'pat' + str(patient_num) + '/MR_T2/'
		T2_volume = load_examples(folder, slices, dim, T2_mean)
		patient_X = np.concatenate((patient_X, T2_volume),1)

		# Load FLAIR images, X.shape =  (num_slices,4dim,PHOTO_WIDTH,PHOTO_WIDTH)
		FLAIR = folder + 'pat' + str(patient_num) + '/MR_FLAIR/'
		FLAIR_volume = load_examples(folder, slices, dim, FLAIR_mean)
		patient_X = np.concatenate((patient_X, FLAIR_volume),1)

		# Load OT labels, Y.shape =  (num_slices,1,PHOTO_WIDTH,PHOTO_WIDTH)
		OT = folder + 'pat' + str(patient_num) + '/OT/'
		OT_volume = load_examples(folder, slices, 1, np.zeros((NUM_SLICES,PHOTO_WIDTH, PHOTO_WIDTH)))
		patient_Y = (OT_volume > 0).astype(int)

		# Add to examples
		X = np.concatenate((X,patient_X),0)
		Y = np.concatenate((Y,patient_Y),0)


def build_net(filter_size = 5, num_channels = NUM_MODALITIES, num_classes = 2, 
				input_var = None):
	network = InputLayer(
			shape=(None, num_channels, PHOTO_WIDTH, PHOTO_WIDTH),
			input_var=input_var)
	network = ConvLayer(network, num_filters=72, filter_size=filter_size)
	network = NormLayer(network)
	network = ConvLayer(DropoutLayer(network, p=0.5), num_filters=256, filter_size=3)
	network = ConvLayer(DropoutLayer(network, p=0.5), num_filters=256, filter_size=3)
	network = ConvLayer(DropoutLayer(network, p=0.5), num_filters=num_classes,
						filter_size=3)
	return network

def train_net(folder, train_set, validation_set, test_set, edge_len, 
				T1_mean, T1c_mean, T2_mean, FLAIR_mean, num_epochs = 500, 
				l1_reg = 0, l2_reg = 0, learn_rate = 0.001, num_classes = 2):
	'''
	Note: The following setup code is adapted from the mnist.py code from lasagne examples.
	The testing structure was changed to match my data setup
	'''
	
	# Set up Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.itensor3('targets')

	# Create neural network
	network = build_net(input_var=input_var)

	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(network)
	loss = ImageCrossEntropyLoss(prediction, target_var,num_classes) + l1_reg*lasagne.regularization.regularize_network_params(network, l1) + l2_reg*lasagne.regularization.regularize_network_params(network, l2)
	loss = loss.mean()
	# We could add some weight decay as well here, see lasagne.regularization.

	# Create update expressions for training, i.e., how to modify the
	# parameters at each training step. Here, we'll use Stochastic Gradient
	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.adam(loss, params, learning_rate=learn_rate)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = ImageCrossEntropyLoss(prediction, target_var,num_classes) + l1_reg*lasagne.regularization.regularize_network_params(network, l1) + l2_reg*lasagne.regularization.regularize_network_params(network, l2)
	test_loss = test_loss.mean()
	# As a bonus, also create an expression for the classification accuracy:
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
						dtype=theano.config.floatX)

	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	# Add a function that computes training accuracy
	train_acc_fn = theano.function([input_var, target_var], test_acc)

	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	'''
	TODO: Change these to reasonable numbers after testing
	'''
	patients_per_batch = 1
	slices_per_patient = 1
	iterations_per_patient = 1
	num_slices = patients_per_batch * slices_per_patient

	print("Starting training...")
	for epoch in range(num_epochs):
		# Draw randomly from training and validation sets
		train_index = map(int,np.floor(np.random.rand(patients_per_batch)*np.asarray(train_set).shape[0]))
		train_batch = [train_set[i] for i in train_index]

		val_index = map(int,np.floor(np.random.rand(1)*np.asarray(train_set).shape[0]))
		val_batch = [train_set[i] for i in train_index]

		print("Starting epoch " + str(epoch))
		train_err = 0
		train_batches = 0
		start_time = time.time()
		# "Full pass" over patients
		for patient in train_batch:
			for i in range(iterations_per_patient):
				inputs, targets = load_data(folder, np.asarray([patient]), edge_len, num_slices, 
											T1_mean, T1c_mean, T2_mean, FLAIR_mean)
				#pdb.set_trace()
				assert len(inputs) == len(targets)
				print("Made it past data loading")
				train_err += train_fn(inputs, targets)
				train_batches += 1

        # And a "full pass" over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        print 'Validating...'
        for patient in val_batch:
        	for i in range(iterations_per_patient):
	            inputs, targets = load_data(folder, np.asarray([patient]), edge_len, num_slices, 
	            							T1_mean, T1c_mean, T2_mean, FLAIR_mean)
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
			inputs, targets = load_data(folder, np.asarray([patient]), edge_len, num_slices, 
										T1_mean, T1c_mean, T2_mean, FLAIR_mean)
			err, acc = val_fn(inputs, targets)
			test_err += err
			test_acc += acc
			test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("  test accuracy:\t\t{:.2f} %".format(
		test_acc / test_batches * 100))

	return val_acc/val_batches, test_acc/test_batches


def main(num_epochs=25,percent_validation=0.05,percent_test=0.10,edge_len=33,
			num_regularization_params = 10):
	rng_state = np.random.get_state()

	if(platform.system() == 'Darwin'):
		folder = '/Users/dominicdelgado/Documents/Radiogenomics/bratsHGG/jpeg/'
	else:
		folder = '/home/ubuntu/Neural_CNN/jpeg/'
		#folder = '/home/ubuntu/data/jpeg/'

	# Calculate mean images
	T1_mean, T1c_mean, T2_mean, FLAIR_mean = get_all_mean_volumes(folder)

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
	# l1_reg = 10**(-1*(np.random.rand(num_regularization_params)*4 + 3))
	# l2_reg = 10**(-1*(np.random.rand(num_regularization_params)*4 + 3))
	# lr = 10 ** (-1 * (np.random.rand(num_regularization_params) * 3 + 3.5))
	# lr = lr.astype(np.float32)
	l1_reg = np.asarray([0])
	l2_reg = np.asarray([0])
	lr = np.asarray([0.001])
	lr = lr.astype(np.float32)
	num_epochs = 1

	best_l1 = l1_reg[0]
	best_l2 = l2_reg[0]
	best_test_pct = 0
	best_val_pct = 0
	best_lr = 0
	data_valid = False

	# Train network
	for i in range(l2_reg.shape[0]):
		val_pct, test_pct = train_net(folder = folder, train_set=train_set, 
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
			data_valid = True

	# Report results and save

	pdb.set_trace()

	print "Achieved test error of %f with l1 = %f, l2 = %f, learn rate = %f." % (test_pct, l1_reg[i], l2_reg[i], lr[i])
	print "Best so far: %f with l1 = %f, l2 = %f, learn rate = %f." % (best_test_pct, best_l1, best_l2, best_lr)

	return 0

if __name__ == '__main__':
	kwargs = {}
	if len(sys.argv) > 1:
		kwargs['num_epochs'] = int(sys.argv[1])
	main(**kwargs)
