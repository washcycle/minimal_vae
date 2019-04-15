
#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input, Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.backend import random_normal
from tensorflow.keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

#%% Data Prep
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz') 
x_train = x_train.reshape(-1,28,28,1)
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape(-1,28,28,1)
x_test = x_test.astype('float32') / 255
image_size = x_train.shape[1]


#%%
class PrintDecodedImage(tf.keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, img):
        self.batch_idx = 0    
        self.img  = img  
        self.batch_idx=0
        
        self.batch_imgs = []

    def on_batch_begin(self, batch, logs=None):

        return super().on_batch_begin(batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):

        self.batch_idx = 0
        self.batch_imgs = []

        return super().on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        nb_imgs = len(self.batch_imgs)
        
        f, ax = plt.subplots(1, nb_imgs)
        for i, _img in enumerate(self.batch_imgs):
            ax[i].imshow(_img.squeeze(), clim=[0, 1], cmap='bone')
        plt.show()
            

        return super().on_epoch_end(epoch, logs=logs)

    def on_batch_end(self, batch, logs={}):
        self.batch_idx += 1

        if(self.batch_idx % 100 == 0):
            decoded_img = self.model.predict(x = self.img)
            self.batch_imgs.append(decoded_img)
                      
        

#%%
# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#%% kl Loss

def calculate_kl_loss(mu, sigma, kl_tolerance = 0.2):
    """ Function to calculate the KL loss term. 
        Considers the tolerance value for which optimization for KL should stop 
    """
    # kullback Leibler loss between normal distributions
    
    kl_cost = -0.5 * tf.keras.backend.mean(1 + sigma - tf.keras.backend.square(mu) - tf.keras.backend.exp(sigma))
    
    return tf.keras.backend.max([kl_cost, kl_tolerance])

#%% Encoder
tf.keras.backend.clear_session()

latent_dim = 10

inputs = Input(shape = (28, 28,1), name='encoder_input')

# Encoder
c1 = Conv2D(filters = 64, kernel_size = 3, strides=2, activation='relu', padding='same')(inputs)
c2 = Conv2D(filters = 128, kernel_size = 3, strides=2, activation='relu', padding='same')(c1)
c3 = Conv2D(filters = 256, kernel_size = 3, strides=2, activation='relu', padding='same')(c2)

x = Flatten()(c3)

mu = Dense(latent_dim, name='mu')(x)
sigma = Dense(latent_dim, name='sigma')(x)

# Sampling

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([mu, sigma])

encoder = Model(inputs = inputs, outputs = [mu, sigma, z], name='encoder')
encoder.summary()

#%% Decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
decoder_input = Dense(4096)(latent_inputs)
decoder_tensor_input = Reshape((4, 4, 256))(decoder_input)
d1 = Conv2DTranspose(filters = 128, kernel_size = 5, strides=2, padding = 'same', name='deconvol_layer_1')(decoder_tensor_input)
d2 = Conv2DTranspose(filters = 64, kernel_size = 5, strides=2, padding = 'same', name='deconvol_layer_2')(d1)
d3 = Conv2DTranspose(filters = 1, kernel_size = 5, strides=2, padding = 'same', name='deconvol_layer_3')(d2)

d4 = Lambda(lambda x: x[:, 2:-2, 2:-2, :], output_shape=(28,28,1))(d3)

outputs = Conv2D(1, (1,1), padding="same", activation="sigmoid")(d4)

decoder = Model(inputs = latent_inputs, outputs = outputs, name='dencoder')
decoder.summary()

#%%
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

#%%
reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
reconstruction_loss *= image_size * image_size
kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

#%%
test_img = x_test[0,:,:,:].reshape([-1, 28, 28, 1])
plt.figure('test image')
plt.imshow(test_img.squeeze(), clim=[0, 1], cmap='bone')
printDecodedImage=PrintDecodedImage(test_img)
vae.fit(x_train, epochs=5, batch_size=128, callbacks=[printDecodedImage],
        validation_data=(x_test, None), verbose=0)

#%%



        
