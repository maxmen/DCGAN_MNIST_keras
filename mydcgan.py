from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from PIL import Image
import argparse
import math

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

class GAN(object):

	def __init__(self,input_dim=100,dim=7,depth_gen=128,depth_disc=64,dropout_gen=0.4,dropout_disc=0.4,\
		ishape=(28,28,3),ch=3,momentum=0.9,lrelu=0.2):
		self.G=self.generator(input_dim=input_dim,dim=dim,depth=depth_gen,dropout=dropout_gen,ch=ch,momentum=momentum,lrelu=lrelu)
		self.D=self.discriminator(depth=depth_disc,dropout=dropout_disc,ishape=ishape)
		self.G.compile(loss='binary_crossentropy', optimizer=Adam())
		self.D.compile(loss='binary_crossentropy', optimizer=Adam())
		self.AD=self.adversarial()

	def generator(self,input_dim=100,dim=7,depth=128,dropout=0.4,ch=3,momentum=0.9,lrelu=0.2):
		G=Sequential()
		G.add(Dense(dim*dim*depth, input_dim=input_dim))
		G.add(BatchNormalization())
		G.add(Activation(LeakyReLU(lrelu)))
		G.add(Reshape((dim, dim, depth)))
		G.add(Dropout(dropout))
		G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
		G.add(BatchNormalization(momentum=momentum))
		G.add(Activation(LeakyReLU(lrelu)))
		G.add(UpSampling2D())
		# 2nd convolutional layer: input [2*dim,2*dim,depth/2], output [4*dim, 4*dim, depth/4]
		G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
		G.add(BatchNormalization(momentum=momentum))
		G.add(Activation(LeakyReLU(lrelu)))
		G.add(UpSampling2D())
		# 3rd convolutional layer: input [4*dim,4*dim,depth/4], output [4*dim, 4*dim, depth/8]
		G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
		G.add(BatchNormalization(momentum=0.9))
		G.add(Activation(LeakyReLU(0.2)))
		# output layer (1,4*dim,4*dim,ch)
		G.add(Conv2DTranspose(ch, 5, padding='same'))
		G.add(Activation('tanh'))
		G.summary()
		return(G)

	def discriminator(self,depth=64,dropout=0.4,ishape=(28,28,3)):
		D = Sequential()
		D.add(Conv2D(depth, 5, strides=2, input_shape=ishape,\
			padding='same'))
		D.add(LeakyReLU(alpha=0.2))
		D.add(Dropout(dropout))
		# 2nd layer: [14,14,128], out [28,28,256]
		D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
		D.add(LeakyReLU(alpha=0.2))
		D.add(Dropout(dropout))
		# 3rd layer
		D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
		D.add(LeakyReLU(alpha=0.2))
		D.add(Dropout(dropout))
		# 4th layer
		D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
		D.add(LeakyReLU(alpha=0.2))
		D.add(Dropout(dropout))
		D.add(Flatten())
		D.add(Dense(1))
		D.add(Activation('tanh'))
		D.summary()
		return (D)

	def adversarial(self):
		optimizer = RMSprop(lr=0.0001, decay=3e-8)
		AD = Sequential()
		AD.add(self.G)
		AD.add(self.D)
		AD.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
		return(AD)

	def pre_train_discriminator(self,Xtrain,Ytrain,batch_size=128,epoch=1):
		batch_count = Xtrain.shape[0] // batch_size
		for i in range(epoch):
			for j in tqdm(range(batch_count)):
				#noise_input = np.random.rand(batch_size, 100)
				noise_input = np.random.uniform(-1, 1, size=(batch_size, 100))
				image_batch = Xtrain[np.random.randint(0, Xtrain.shape[0], size=batch_size)]
				predictions = self.G.predict(noise_input, batch_size=batch_size)
				X = np.concatenate([predictions, image_batch])
				y_discriminator = [0]*batch_size + [1]*batch_size
				self.D.trainable = True
				self.D.train_on_batch(X, y_discriminator)

	def train(self,Xtrain,Ytrain,batch_size=128,epoch=50):
		batch_count = Xtrain.shape[0] // batch_size
		for i in range(epoch):
			for j in tqdm(range(batch_count)):
				#noise_input = np.random.rand(batch_size, 100)
				noise_input = np.random.uniform(-1, 1, size=(batch_size, 100))
				image_batch = Xtrain[np.random.randint(0, Xtrain.shape[0], size=batch_size)]
				predictions = self.G.predict(noise_input, batch_size=batch_size)
				if j % 20 == 0:
					image = combine_images(predictions)
					image = image*127.5+127.5
					Image.fromarray(image.astype(np.uint8)).save(str(i)+"_"+str(j)+".png")
				X = np.concatenate([predictions, image_batch])
				y_discriminator = [0]*batch_size + [1]*batch_size
				self.D.trainable = True
				D_loss=self.D.train_on_batch(X, y_discriminator)
				print("batch %d D_loss : %f" % (j, D_loss))
				noise_input = np.random.rand(batch_size, 100)
				y_generator = [1]*batch_size
				self.D.trainable = False
				AD_loss=self.AD.train_on_batch(noise_input, y_generator)
				print("batch %d AD_loss : %f" % (j, AD_loss))
				if j % 10 == 9:
					self.G.save_weights('generator_weights', True)
					self.D.save_weights('discriminator_weights', True)



if __name__ == "__main__":
	(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
	Xtrain = (Xtrain - 127.5) / 127.5
	Xtest = (Xtest - 127.5) / 127.5
	Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2], 1)
	Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1)
	Xtrain = Xtrain.astype('float32')

	GAN=GAN(ch=1,ishape=(28,28,1))
	GAN.pre_train_discriminator(Xtrain,Ytrain,epoch=1)
	GAN.train(Xtrain,Ytrain)