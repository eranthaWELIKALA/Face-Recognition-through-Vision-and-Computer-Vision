from library import *
import cv2
import os
from scipy import misc
import numpy as np

def read_data(path):
	data = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		for dirname in dirnames:
			for f in os.listdir(dirpath + dirname):
				try:
				    img = np.ravel(misc.imread(dirpath + dirname + '/' + f, flatten=True))/255
				    data.append((dirname, img))
				except:
					pass
	return data

imgs = np.array([])
print(imgs.shape)
print(np.append([2,3,4],[1,2,3]))
print(imgs)
lbls = []
for (dirpath, dirnames, filenames) in os.walk("images/"):
		for dirname in dirnames:
			for f in os.listdir(dirpath + dirname):
				#print(f,dirname)
				image = cv2.imread("images/"+dirname+"/"+f)
				image = cv2.resize(image, (240, 360)) 
				print(image.shape)
				np.append(imgs,image)
				lbls.append(dirname)

labels = np.array(lbls)

print(imgs)

# img = misc.imread("images/abc/abc0.jpg")
# print(img)

# X_train, images2, Y_train, labels2 = train_test_split(images, labels, test_size=0.4, random_state=0)
# X_validate, X_test, Y_validate, Y_test = train_test_split(images2, labels2, test_size=0.5, random_state=0)


