import numpy as np
import cv2
import os
from sklearn import svm, linear_model, metrics
from database_creation import *
from features_extraction import *
import time


WINDOW_HEIGHT 	= 128;
WINDOW_WIDTH	= 64;

def detect(image_path, classifier):
	start = time.time();
	image = cv2.imread(image_path);
	height, width, channels = np.shape(image);
	step = 8;
	
	y_pad = (height%step)//2;
	x_pad = (width%step)//2;
	
	x_end = width - WINDOW_WIDTH - step;
	y_end = height - WINDOW_HEIGHT - step;
	
	detected = [];
	n_detected = 0;
	n_rep = 0;
	
	for y in range(y_pad, y_end, step):
		for x in range(x_pad, x_end, step):
			cropped = image[y:y+WINDOW_HEIGHT,x:x+WINDOW_WIDTH];
			features = extractFeatures(cropped);
			prediction = classifier.predict([features]);
			n_rep+=1;
			if(prediction==1):
				detected.append([x,y]);
				n_detected +=1;
	
	end = time.time();
	elapsed = end - start;
	
	print("Detected %d windows in %d seconds and %d rep"%(n_detected, elapsed, n_rep));
	
	for i in range(n_detected):
		x = detected[i][0];
		y = detected[i][1];
		cv2.rectangle(image, (x,y), (x+WINDOW_WIDTH,y+WINDOW_HEIGHT), (255,0,0), 2);
	
	cv2.imshow("windows", image);
	cv2.waitKey(0);
	cv2.destroyAllWindows();
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	