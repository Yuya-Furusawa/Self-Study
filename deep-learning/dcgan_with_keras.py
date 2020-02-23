import sys
import os
sys.path.append('/usr/local/lib/python3.7/dist-packages')

import keras
from keras import layers
from keras import datasets
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.initializers import TruncatedNormal

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

def load_data(path):
    img_size = (128,128)

    dir_list = ['00000', '01000', '02000', '03000', '04000', '05000', '06000',
                '07000', '08000', '09000', '10000', '11000', '12000', '13000',
                '14000', '15000', '16000', '17000', '18000', '19000', '20000',
                '21000', '22000', '23000', '24000', '25000', '26000', '27000',
                '28000', '29000']

    temp_img_array_list = []

    i = 0
    for directory in dir_list:
        img_list = glob.glob(path + directory + '/*.png')
        for img in img_list:
            temp_img = load_img(img, grayscale=False, target_size=img_size)
            temp_img_array = img_to_array(temp_img) /255
            temp_img_array_list.append(temp_img_array)

    temp_img_array_list = np.array(temp_img_array_list)

    return temp_img_array_list


latent_dim = 100
height = 128
width = 128
channels = 3

initializer = TruncatedNormal(stddev=0.02)

#= generator =#

generator_input = keras.Input(shape=(latent_dim,))

x = layers.Dense(8*8*1024)(generator_input)
x = layers.normalization.BatchNormalization(epsilon=0.00005)(x)
x = layers.Reshape((8, 8, 1024))(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(512, 5, strides=2, padding='same', kernel_initializer=initializer)(x)
x = layers.normalization.BatchNormalization(epsilon=0.00005)(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 5, strides=2, padding='same', kernel_initializer=initializer)(x)
x = layers.normalization.BatchNormalization(epsilon=0.00005)(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(128, 5, strides=2, padding='same', kernel_initializer=initializer)(x)
x = layers.normalization.BatchNormalization(epsilon=0.00005)(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(64, 5, strides=2, padding='same', kernel_initializer=initializer)(x)
x = layers.normalization.BatchNormalization(epsilon=0.00005)(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(channels, 5, strides=1, activation='tanh', padding='same', kernel_initializer=initializer)(x)

generator = keras.models.Model(inputs=generator_input, outputs=x)


#= discriminator =#

discriminator_input = layers.Input(shape=(height, width, channels))

x = layers.Conv2D(64, 5, strides=2, kernel_initializer=initializer)(discriminator_input)
x = layers.normalization.BatchNormalization(epsilon=0.00005)(x)
x = layers.LeakyReLU(alpha=0.2)(x)

x = layers.Conv2D(128, 5, strides=2, padding='same', kernel_initializer=initializer)(x)
x = layers.normalization.BatchNormalization(epsilon=0.00005)(x)
x = layers.LeakyReLU(alpha=0.2)(x)

x = layers.Conv2D(256, 5, strides=2, padding='same', kernel_initializer=initializer)(x)
x = layers.normalization.BatchNormalization(epsilon=0.00005)(x)
x = layers.LeakyReLU(alpha=0.2)(x)

x = layers.Conv2D(512, 5, strides=1, padding='same', kernel_initializer=initializer)(x)
x = layers.normalization.BatchNormalization(epsilon=0.00005)(x)
x = layers.LeakyReLU(alpha=0.2)(x)

x = layers.Conv2D(1024, 5, strides=2, padding='same', kernel_initializer=initializer)(x)
x = layers.normalization.BatchNormalization(epsilon=0.00005)(x)
x = layers.LeakyReLU(alpha=0.2)(x)

x = layers.Flatten()(x)
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(inputs=discriminator_input, outputs=x)

discriminator_optimizer = keras.optimizers.Adam(0.00004, 0.5)
discriminator.compile(optimizer=discriminator_optimizer,
                      loss='binary_crossentropy')


#= GAN =#

discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(inputs=gan_input, outputs=gan_output)
gan_optimizer = keras.optimizers.Adam(0.0002, 0.5)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


def save_images(epoch, generator):
    #Save 25 generated images for demonstration purposes using matplotlib.pyplot.
    rows, columns = 5, 5
    noise = np.random.uniform(-1, 1, (rows * columns, 100))
    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5

    figure, axis = plt.subplots(rows, columns)
    image_count = 0
    for row in range(rows):
        for column in range(columns):
            axis[row,column].imshow(generated_images[image_count, :], cmap='spring')
            axis[row,column].axis('off')
            image_count += 1
    figure.savefig("gan_images/generated_%d.png" % epoch)
    plt.close()


#= Training =#

#setup directory
image_dir = '/home/takatom/GAN/thumbnails128x128/'
save_dir = '/home/takatom/GAN/gan_images/'

#read data
x_train = load_data(image_dir)

#normalized data
x_train = (x_train.astype(np.float32) - 127.5) / 127.5

iterations = 50000
batch_size = 64

start = 0
for step in range(iterations):
    #generate noise
    random_latent_vectors = np.random.uniform(-1, 1, size=(batch_size, latent_dim))

    #generate fake image
    generated_images = generator.predict(random_latent_vectors)

    #combine fake image and real image
    stop = start + batch_size
    real_images = x_train[start:stop]
    combined_images = np.concatenate([real_images, generated_images])

    #create label(1=real, 0=fake) with noise
    labels = np.concatenate([0.5 * np.random.random_sample((batch_size, 1)) + 0.7,
                             0.3 * np.random.random_sample((batch_size, 1))])

    #train discriminator
    d_loss = discriminator.train_on_batch(combined_images, labels)

    #generate noise
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    #create label(1=true)
    misleading_targets = np.ones((batch_size, 1))

    #train gan
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0

    #save images
    if step % 1000 == 0:
        gan.save_weights('gan.h5')

        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))

        save_images(step, generator)
