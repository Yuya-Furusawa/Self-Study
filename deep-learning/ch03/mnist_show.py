import sys, os
sys.path.append("/Users/yuyafurusawa/deep-learning-from-scratch/dataset/")
from mnist import load_mnist
from PIL import Image
import numpy as np

def img_show(img):
	pil_img = Image.fromarray(np.uint8(img))
	pil_img.show()

(x_train, t_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)            # (784,)
img = img.reshape(28, 28)   # Reshape to original size
print(img.shape)            # (28, 28)

img_show(img)
