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


(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

Xtrain = (Xtrain - 127.5) / 127.5
Xtest = (Xtest - 127.5) / 127.5


Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2], 1)
Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1)

Xtrain = Xtrain.astype('float32')


N=100

def samplePZ(batch_size=1):
    z = np.random.uniform(-1, 1, [batch_size,N])
    return(z)



depth=128
dim=int(Xtrain.shape[1]/4)
dropout = 0.4

def generator(input_dim=100,dim=7,depth=128,droput=0.4,ch=3,momentum=0.9,lrelu=0.2):
    G=Sequential()
    # Input layer [1,100]
    G.add(Dense(dim*dim*depth, input_dim=input_dim))
    G.add(BatchNormalization())
    G.add(Activation(LeakyReLU(lrelu)))
    G.add(Reshape((dim, dim, depth)))
    G.add(Dropout(dropout)) #setting a fraction rate of input units to 0 at each update during training time, 
                            #which helps prevent overfitting.
    
    # 1st convolutional layer: input [dim,dim,depth], output [2*dim, 2*dim, depth/2]
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
    
G=generator(ch=1) 
G.compile(loss='binary_crossentropy', optimizer=Adam())

def discriminator(depth=64,dropout=0.4,ishape=(28,28,3)):
    D = Sequential()
    # 1st layer: in [ishape], out [14,14,128]
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
    return D

D=discriminator(ishape=(28,28,1))
D.compile(loss='binary_crossentropy', optimizer=Adam())

def adversarial():
    optimizer = RMSprop(lr=0.0001, decay=3e-8)
    AD = Sequential()
    AD.add(generator())
    AD.add(discriminator())
    AD.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
    return AD

AD=adversarial()


batch_size=128
epoch=1
batch_count = Xtrain.shape[0] // batch_size
for i in range(epoch):
    for j in tqdm(range(batch_count)):
        noise_input = np.random.rand(batch_size, 100)
        image_batch = Xtrain[np.random.randint(0, Xtrain.shape[0], size=batch_size)]
        predictions = G.predict(noise_input, batch_size=batch_size)
        X = np.concatenate([predictions, image_batch])
        y_discriminator = [0]*batch_size + [1]*batch_size
        D.trainable = True
        D.train_on_batch(X, y_discriminator)


batch_size=128
epoch=5
batch_count = Xtrain.shape[0] // batch_size
loss=[]
for i in range(epoch):
    for j in tqdm(range(batch_count)):
        # the first part of the look is identical to training D
        noise_input = np.random.rand(batch_size, 100)
        image_batch = Xtrain[np.random.randint(0, Xtrain.shape[0], size=batch_size)]
        predictions = G.predict(noise_input, batch_size=batch_size)
        X = np.concatenate([predictions, image_batch])
        y_discriminator = [0]*batch_size + [1]*batch_size
        D.trainable = True
        D.train_on_batch(X, y_discriminator)
        # now we set the Discriminator as untrainable and we generate mock images trying to maximize the rate 
        # of images that the discriminator classifies as false.
        # Note that, at each step, the discriminator is updated (as shown in the previous lines)
        noise_input = np.random.rand(batch_size, 100)
        # set labels of generated images 
        y_generator = [1]*batch_size
        D.trainable = False
        # train the adversarial model
        loss.append(AD.train_on_batch(noise_input, y_generator))

steps=np.linspace(0,len(loss),len(loss))

plt.plot(steps,loss)

def plot_output():
    try_input = np.random.rand(100, 100)
    preds = G.predict(try_input)

    plt.figure(figsize=(10,10))
    for i in range(preds.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(preds[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.show()
    
    # tight_layout minimizes the overlap between 2 sub-plots
    #plt.tight_layout()
    
plot_output()