import os
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

"""
データ拡張を行う特徴抽出
- VGG16 modelをつかって特徴を抽出し、それを全結合分類器に渡す
"""

#use VGG16 model
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

#setup directories
base_dir = '/Users/yuyafurusawa/Downloads/cats_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

#costruct model
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#VGG16 modelのweightは更新しない
conv_base.trainable = False

train_datagen = ImageDataGenerator(rescale=1./255,
								   rotation_range=40,
								   width_shift_range=0.2,
								   height_shift_range=0.2,
								   shear_range=0.2,
								   zoom_range=0.2,
								   horizontal_flip=True,
								   fill_model='nearest')
#validation dataは拡張しない
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
													target_size=(150,150),
													batch_size=20,
													class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
														target_size=(150,150),
														batch_size=20,
														class_mode='binary')

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
			  loss='binary_crossentropy',
			  metrics=['acc'])

history = model.fit_generator(train_generator,
							  steps_per_epoch=100,
							  epochs=30,
							  validation_data=validation_generator,
							  validation_steps=50,
							  verbose=2)

