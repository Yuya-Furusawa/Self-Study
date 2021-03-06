import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.datasets import boston_housing

# load data
(train_data, train_targets), (test_data, test_targets) = \
	boston_housing.load_data()

# normalize data
mean = train_data.mean(axis=0)
train_data -= mean
test_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data /= std

def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	return model

# j-fold cross validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
	print('processing fold #', i)

	# i-th fold data : validation data
	val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
	val_targets = train_targets[i * num_val_samples : (i+1) * num_val_samples]

	# training data
	partial_train_data = np.concatenate(
		[train_data[: i * num_val_samples], train_data[(i+1) * num_val_samples :]],
		axis=0)
	partial_train_targets = np.concatenate(
		[train_targets[: i * num_val_samples], train_targets[(i+1) * num_val_samples :]])

	# construct model
	model = build_model()

	# fit model
	model.fit(partial_train_data, partial_train_targets,
			  epochs=num_epochs, batch_size=1, verbose=0)

	# evaluate model
	val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
	all_scores.append(val_mae)
