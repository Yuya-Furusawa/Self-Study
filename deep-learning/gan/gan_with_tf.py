import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutrials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')

def model_inputs(real_dim, z_dim):
	inputs_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real')
	inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

	return inputs_real, inputs_z

def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
	with tf.variable_scope('generator', reuse=reuse):
		h1 = tf.layers.dense(z, n_units, activation=None)
		h1 = tf.maximum(alpha * h1, h1) #Leaky ReLU

		logits = tf.layers.dense(h1, out_dim, activation=None)
		out = tf.tanh(logits)

		return out

def discriminator(x, n_units=128, reuse=False, alpha=0.01):
	with tf.variable_scope('discriminator', reuse=reuse):
		h1 = tf.layers.dense(x, n_units, activation=None)
		h1 = tf.maximum(alpha * h1, h1)

		logits = tf.layers.dense(h1, 1, activation=None)
		out = tf.sigmoid(logits)

		return out, logits


# initialize hyperparameters
input_size = 784
z_size = 100
g_hidden_size = 128
d_hidden_size = 128
alpha = 0.01
smooth = 0.1

