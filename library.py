import cv2
import os
import matplotlib as plt


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


def convert2HSV(img):
	# Gets a RGB image an output a HSV image
	return cv2.cvtColor(img, cv2.COLOR_RGB2HSI)


def reduce_noise(img):
	# Returns the noise reduced image
	return cv2.fastNlMeansDenoisingColored(img)



#split_video("abc.mp4")

# for f in os.listdir("images"):
# 	for img in os.listdir("images/"+f):
# 		print(img)

#image = convert2HSV('images/abc/frame0.jpg')
#cv2.imshow('image',img)
# for i in image:
# 	print(i)
# cv2.imshow('image',image)
# cv2.waitKey(0)
img = cv2.imread('images/abc/frame0.jpg')
im2 = reduce_noise(img)
cv2.imshow('image1',img)
cv2.imshow('image2',im2)
cv2.waitKey(0)