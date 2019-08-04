from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

"""
2-dimensional Convolution Layer
Conv2D(
       filters, #出力空間の次元(出力filterの数)
	   kernel_size, #畳み込みwindowのwidth&height
	   strides=(1,1), #strideの大きさ
	   padding='valid', #padding
       activation=None #activation function
       input_shape, #第1層目で使用する場合に指定
       )

2-dimensional Pooling Layer
MaxPooling2D(
             pool_size=(2,2), #down scaleする係数, (2,2)だと縦と横それぞれ大きさが半分になる
             strides=None,
             padding='valid'
             )
"""

# create neural network
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# training
(train_images, train_labels), (test_images, test_labels) = \
    mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)