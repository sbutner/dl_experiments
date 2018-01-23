import keras
import matplotlib

from keras.layers import Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D, Input, add
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, load_model
from keras.datasets import cifar100
from matplotlib.image import imsave


import numpy as np

# function definitions

def add_residual_block(x):
	"""
	
	Adds a residual block comprising a 1x1 convolution, a 3x3 convolution, then a 1x1 convolution. Follows Newell, et al., (2016) in restricting the feature map depth. 
	
	Args:
		x->Tensor: output of previous layer(s)
		
	"""
	_skip = x
	x = Conv2D(16, kernel_size = (1,1), padding = 'same', data_format = 'channels_first', activation = 'relu')(x)
	x = BatchNormalization(axis = 1)(x)
	x = Conv2D(16, kernel_size = (3,3), padding = 'same', data_format = 'channels_first', activation = 'relu')(x)
	x = BatchNormalization(axis = 1)(x)
	x = Conv2D(32, kernel_size = (1,1), padding = 'same', data_format = 'channels_first', activation = 'relu')(x)
	x = BatchNormalization(axis = 1)(x)
	x = keras.layers.add([x, _skip])
	
	return x

def print_image(cifar_array, path, file):
	"""
	Saves a PNG of the passed in RGB, H, W array from the Keras datasets loader
	
	Args:
		cifar_array: numpy array in channels_first format.
		path: OS path for individual files
		file: name for the individual file
	"""
	cifar_array = np.transpose(cifar_array, (1,2,0))
	return matplotlib.image.imsave(path+file+".png", cifar_array)

def normalize(numpy_array, new_max = 1, new_min = 0):
	"""
	Return a rescaled numpy array
	
	TODO: add remainder of formula for rescaling to ranges other than 0..1
	"""
	
	min = np.amin(numpy_array)
	max = np.amax(numpy_array)
	
	numpy_array = (numpy_array - min) / (max - min)
	
	return numpy_array

def build_autoencoder():
	"""
	Assembles the keras Model object.
	
	Uses an hourglass style encoder-decoder with a bottleneck of 16-dims (8% compression)
	"""

	input_image = Input(shape=(3,32,32))

	x = Conv2D(32, kernel_size = (1,1), padding = 'same', data_format = 'channels_first', activation = 'relu')(input_image)
	_skip = x
	x = add_residual_block(x) 
	x = MaxPooling2D((2,2), padding = 'same')(x) #16x16x256
	x = add_residual_block(x) 
	x = MaxPooling2D((2,2), padding = 'same')(x) #8x8x256
	x = add_residual_block(x) 
	x = MaxPooling2D((2,2), padding = 'same')(x) #4x4x256
	x = add_residual_block(x) 
	x = MaxPooling2D((2,2), padding = 'same')(x) #2x2x256
	x = add_residual_block(x) 
	x = MaxPooling2D((2,2), padding = 'same')(x) #1x1x256

	encoded = add_residual_block(x) # 8% of original size

	x = add_residual_block(encoded)
	x = UpSampling2D((2,2))(x) #2x2x256
	x = add_residual_block(x)
	x = UpSampling2D((2,2))(x) #4x4x256
	x = add_residual_block(x)
	x = UpSampling2D((2,2))(x) #8x8x256
	x = add_residual_block(x)
	x = UpSampling2D((2,2))(x) #16x16x256
	x = add_residual_block(x)
	x = UpSampling2D((2,2))(x) #32x32x256
	x = keras.layers.add([x, _skip])

	decoded = Conv2D(3, (1,1), activation = 'relu', data_format = 'channels_first', padding = 'same')(x)

	return Model(input_image, decoded)

	
# Data Prep
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Model Building

try:
	autoencoder = load_model('weights.hdf5')
except:
	autoencoder = build_autoencoder()
	autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

datagen = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True, rotation_range = 20, width_shift_range = 0.2, height_shift_range = 0.2, horizontal_flip = True)
datagen.fit(x_train)

for i in range(10):
	autoencoder.fit_generator(datagen.flow(x_train, x_train, batch_size = 128), steps_per_epoch = 7, epochs = 10,  callbacks = [TensorBoard(log_dir='/tmp/autoencoder'), ModelCheckpoint('weights.hdf5', monitor = 'loss', verbose = 1, save_best_only = True, mode = 'max'), ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)])


	#evaluate and print

	preview_batch = next(datagen.flow(x_train, batch_size = 1))
	preview_batch_predictions = autoencoder.predict_on_batch(preview_batch)

	preview_batch = normalize(preview_batch)
	preview_batch_predictions = normalize(preview_batch_predictions)

	for x, y in zip(preview_batch, preview_batch_predictions):
		print_image(x, '.\\images\\', 'original'+str(i))
		print_image(y, '.\\images\\', 'reconstructed'+str(i))
		i += 1