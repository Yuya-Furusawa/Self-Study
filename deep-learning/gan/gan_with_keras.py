from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import mnist
import numpy as np


class GAN:
	def __init__(self):
		self.img_rows = 28
		self.img_cols = 28
		self.img_channels = 1
		self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

		self.z_dim = 100

		optimizer = optimizers.Adam(0.0002, 0.5)

		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='cross_binaryentropy',
								   optimizer=optimizer,
								   metrics=['accuracy'])

		self.generator = self.build_generator()

		self.combined = self.build_combined1()
		self.combined.compile(loss='cross_binaryentropy', optimizer=optimizer)

	def build_generator(self):
		noise_shape = (self.z_dim,)

		model = models.Sequential()
		model.add(layers.Dense(256, input_shape=self.img_shape))
		model.add(layers.LeakyReLU(alpha=0.2))
		model.add(layers.BatchNormalization(momentum=0.8))
		model.add(layers.Dense(512))
		model.add(layers.LeakyReLU(alpha=0.2))
		model.add(layers.BatchNormalization(momentum=0.8))
		model.add(layers.Dense(1024))
		model.add(layers.LeakyReLU(alpha=0.2))
		model.add(layers.BatchNormalization(momentum=0.8))
		model.add(layers.Dense(np.prod(self.img_shape), activation='tanh'))
		model.add(layers.Reshape(self.img_shape))

		return model

	def build_discriminator(self):
		model = models.Sequential()
		model.add(layers.Dense(512))
		model.add(layers.LeakyReLU(alpha=0.2))
		model.add(layers.Dense(256))
		model.add(layers.LeakyReLU(alpha=0.2))
		model.add(layers.Dense(1, activation='sigmoid'))

		return model

	def build_combined1(self):
		self.discriminator.trainable = False
		model = models.Sequential([self.generator, self.discriminator])

		return model

	def build_combined2(self):
		z = layers.Input(shape=(self.z_dim,))
		img = self.generator(z)
		self.discriminator.trainable = False
		valid = self.discriminator(img)
		model = models.Model(inputs=z, outputs=valid)

		return model

	def train(self, epochs, batch_size=128, save_interval=50):
		(x_train, _), (_, _) = mnist.load_data()

		x_train = (x_train.astype(np.float32) - 127.5) / 127.5
		x_train = np.expand_dims(x_train, axis=3)

		half_batch = int(batch_size / 2)
		num_batches = int(x_train.shape[0] / half_batch)

		for epoch in range(epochs):
			for iteration in range(num_batches):
				noise = np.random.normal(0, 1, (half_batch, self.z_dim))
				gen_imgs = self.generator.predict(noise)

				idx = np.random.randint(0, x_train.shape[0], half_batch)
				imgs = x_train[idx]

				d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
				d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

				noise = np.random.normal(0, 1, (batch_size, self.z_dim))
				valid_y = np.array([1] * batch_size)

				g_loss = self.combined.train_on_batch(noise, valid_y)

				if epoch % save_interval == 0:
					self.save_imgs(epoch)

	def save_imgs(self, epoch):
		r, c = 5, 5

		noise = np.random.normal(0, 1, (r*c, self.z_dim))
		gen_imgs = self.generator.predict(noise)

		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()
