import csv
import cv2
import numpy as np
from scripts.hyperparams import *

lines=[]
with open(csv_path) as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
		lines.append(line)
images=[]
measurements=[]
for line in lines:
	source_path=line[0]
	filename=source_path.split('/')[-1]
	current_path=base_path+filename
	image=cv2.imread(current_path)
	images.append(image)
	measurement=float(line[3])
	measurements.append(measurement)
X_train=np.array(images)
Y_train=np.array(measurements)
height =np.shape(X_train)[1]
width =np.shape(X_train)[2]

if __name__=="__main__":
	print(height,width,Y_train[6])