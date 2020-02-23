import sys
import os
sys.path.append('/usr/local/lib/python3.7/dist-packages')

from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import glob




class GAN:
    def __init__(self):
        #data size
        self.img_rows = 128
        self.img_cols = 128
        self.img_channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        #setup path
        self.image_dir = '/home/takatom/GAN/thumbnails128x128/'
        self.save_dir = '/home/takatom/GAN/gan_images/'

        #dimension of noise
        self.z_dim = 100

        optimizer = optimizers.Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='cross_binaryentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()

        self.combined = self.build_combined()
        self.combined.compile(loss='cross_binaryentropy', optimizer=optimizer)

    def build_generator(self):
        """
        Generator
        """
        noise_shape = (self.z_dim,)

        model = models.Sequential()
        model.add(layers.Dense(128*64*64, input_shape=noise_shape))
        model.add(layers.normalization.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((64, 64, 128)))
        model.add(layers.Conv2D(256, 5, padding='same'))
        model.add(layers.normalization.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(256, 4, strides=2, padding='same'))
        model.add(layers.normalization.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(256, 5, padding='same'))
        model.add(layers.normalization.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(256, 5, padding='same'))
        model.add(layers.normalization.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(self.img_channels, 7, activation='tanh', padding='same'))

        return model

    def build_discriminator(self):
        """
        Discriminator
        """
        model = models.Sequential()
        model.add(layers.Conv2D(128, 3, input_shape=self.img_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(128, 4, strides=2))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(128, 4, strides=2))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(128, 4, strides=2))
        model.add(layers.LeakyReLU())
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def build_combined(self):
        """
        Generator + Discriminator
        """
        self.discriminator.trainable = False
        model = models.Sequential([self.generator, self.discriminator])

        return model

    def load_data(self, path):
        img_size = (128,128)

        dir_list = ['00000', '01000', '02000', '03000', '04000', '05000', '06000', \
                    '07000', '08000', '09000']

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

    def train(self, epochs, batch_size=128, save_interval=1000):
        #read data
        x_train = self.load_data(self.image_dir)

        #normalize data
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            #-----------------------
            # Training Discriminator
            #-----------------------
            noise = np.random.normal(0, 1, (half_batch, self.z_dim))
            gen_imgs = self.generator.predict(noise)

            #train data
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            imgs = x_train[idx]

            #training
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #-------------------
            # Training Generator
            #-------------------
            noise = np.random.normal(0, 1, (batch_size, self.z_dim))
            valid_y = np.array([1] * batch_size)

            g_loss = self.combined.train_on_batch(noise, valid_y)

            if epoch % save_interval == 0:
                print("epoch:%d, [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
                img = array_to_img(generated_images[0] * 255., scale=False)
                img.save(os.path.join(self.save_dir, 'generated_human' + str(step) + '.png'))
