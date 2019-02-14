import os
import struct
import numpy as np

# Load MNIST data from `path`
def load_mnist(path, kind='train'):
	# join paths
	labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
	images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

	# read file
	# label data
	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II', lbpath.read(8))
		labels = np.fromfile(lbpath, dtype=np.uint8)

	# image data
	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
		images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
		images = ((images / 255.) - .5) * 2

	return images, labels