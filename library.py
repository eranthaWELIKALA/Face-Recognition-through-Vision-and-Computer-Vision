import cv2
import os
import matplotlib as plt
from matplotlib import pyplot as plt
import numpy as np

#--------------------------------------------------------------------------------------------

def plotting_images(original, proccessed_image):
	plt.subplot(121),plt.imshow(original),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(proccessed_image),plt.title('Blurred')
	plt.xticks([]), plt.yticks([])
	plt.show()

#--------------------------------------------------------------------------------------------

def show_images(window_name,img):
	cv2.imshow(window_name,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------------

def split_video(video_name_full):
	# Input is the name of the video with the extension
	# Output is the folder name which is containing frames

	root = "images"

	# Splitiig video name
	video = video_name_full.split(".")
	video_name = video[0]
	video_ext = video[1]

	video_capture = cv2.VideoCapture(video_name_full)
	count = 0
	success = True
	dir_path = root+'/'+video_name
	os.mkdir(dir_path)

	while success:
		success, image = video_capture.read()
		print("Read a new frame: %d"%count,success)
		cv2.imwrite(os.path.join(dir_path, "frame%d.jpg"%count), image)
		count += 1

#--------------------------------------------------------------------------------------------

def convert2HSV(img):
	# Gets a RGB image an output a HSV image
	return cv2.cvtColor(img, cv2.COLOR_RGB2HSI)

#--------------------------------------------------------------------------------------------

def reduce_noise(img):
	# Returns the noise reduced image
	return cv2.fastNlMeansDenoisingColored(img)

#--------------------------------------------------------------------------------------------

def add_noise(noise_typ, img):
	# Returns the noise reduced image

	# 'gauss'     Gaussian-distributed additive noise.
	# 'poisson'   Poisson-distributed noise generated from the data.
	# 's&p'       Replaces random pixels with 0 or 1.
	# 'speckle'   Multiplicative noise using out = image + n*image,where
	#		      n is uniform noise with specified mean & variance.

	if noise_typ == "gaussian":
		row,col,ch= img.shape
		mean = 0
		var = 0.1
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = img + gauss
		return noisy
	elif noise_typ == "s&p":
		row,col,ch = img.shape
		s_vs_p = 0.5
		amount = 0.004
		out = np.copy(img)
		# Salt mode
		num_salt = np.ceil(amount * img.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
		out[coords] = 0
		return out
	elif noise_typ == "poisson":
		vals = len(np.unique(img))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(img * vals) / float(vals)
		return noisy
	elif noise_typ =="speckle":
		row,col,ch = img.shape
		gauss = np.random.randn(row,col,ch)
		gauss = gauss.reshape(row,col,ch)        
		noisy = img + img * gauss
		return noisy

#--------------------------------------------------------------------------------------------

def blur_image(img, blur_type, horizontally_drag = 0, vertically_drag = 0, kernal_size = 0, d = 9, sigmaColor = 75, sigmaSpace = 75):
	# Returns the blured image
	if blur_type == "conv":
		# Blur by convolving image with a normalized box filter
		return cv2.blur(img, (horizontally_drag, vertically_drag))
	elif blur_type == "gaussian":
		return cv2.GaussianBlur(img, (horizontally_drag, vertically_drag),0)
	elif blur_type == "median":
		print(kernal_size)
		return cv2.medianBlur(img,kernal_size)
	elif blur_type == "bilateral":
		return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)


#--------------------------------------------------------------------------------------------

img = cv2.imread('images/abc/frame0.jpg')
img2 = add_noise("speckle",img)
plotting_images(img, img2)