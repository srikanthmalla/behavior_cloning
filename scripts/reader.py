import csv
import cv2
import numpy as np
from scripts.hyperparams import *
from PIL import Image

lines=[]
with open(csv_path) as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
		lines.append(line)
car_images=[]
steering_angles=[]
for line in lines:
	img_center = np.asarray(Image.open(base_path + (line[0]).split('/')[-1]))
	img_left = np.asarray(Image.open(base_path + (line[1]).split('/')[-1]))
	img_right = np.asarray(Image.open(base_path + (line[2]).split('/')[-1]))
	# add images and angles to data set
	car_images.extend((img_center, img_left, img_right))
	steering_center=float(line[3])
	# create adjusted steering measurements for the side camera images
	correction = 0.2 # this is a parameter to tune
	steering_left = steering_center + correction
	steering_right = steering_center - correction
	steering_angles.extend((steering_center, steering_left, steering_right))

X_train=np.array(car_images)
Y_train=np.array(steering_angles)
height =np.shape(X_train)[1]
width =np.shape(X_train)[2]

if __name__=="__main__":
	print(height,width,np.shape(X_train),np.shape(Y_train))