import csv
import cv2
import numpy as np

lines=[]
with open('./dataset/driving_log.csv') as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
		lines.append(line)
images=[]
measurements=[]
for line in lines:
	source_path=line[0]
	filename=source_path.split('/')[-1]
	current_path='./dataset/IMG/'+filename
	image=cv2.imread(current_path)
	images.append(image)
	measurement=float(line[3])
	measurements.append(measurement)
X_train=np.array(images)
Y_train=np.array(measurements)

if __name__=="__main__":
	print(np.shape(X_train))
	print(np.shape(Y_train))