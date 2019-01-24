from library import *
import cv2
import os

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
lbls = []
for (dirpath, dirnames, filenames) in os.walk("images/"):
		for dirname in dirnames:
			for f in os.listdir(dirpath + dirname):
				print(f,dirname)
				np.append(imgs,cv2.imread("images/"+dirname+"/"+f))
				lbls.append(dirname)

labels = np.array(lbls)

print(imgs.shape())

# X_train, images2, Y_train, labels2 = train_test_split(images, labels, test_size=0.4, random_state=0)
# X_validate, X_test, Y_validate, Y_test = train_test_split(images2, labels2, test_size=0.5, random_state=0)


