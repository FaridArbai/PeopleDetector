import numpy as np
import cv2
import os
from sklearn import svm, linear_model, metrics
from database_creation import *
from features_extraction import *
import time
from imutils.object_detection import non_max_suppression

WINDOW_HEIGHT 	= 128;
WINDOW_WIDTH	= 64;
SCALE_STEP = 1.2;
WINDOW_STEP = 8;

hog = cv2.HOGDescriptor();

def detect(orig_image, classifier):
	start = time.time();
	height, width, channels = np.shape(orig_image);
	
	pyramid = [orig_image]
	windows = []
	dims = (int(width//SCALE_STEP), int(height//SCALE_STEP))
	
	while (dims[0]>128 and dims[1]>64):
		blurred = cv2.GaussianBlur(pyramid[-1], ksize=(5, 5), sigmaX=0.6)
		pyramid.append(cv2.resize(src=blurred, dsize=dims))
		dims = (int(dims[0]//1.2), int(dims[1]//1.2))
	
	scale = 1
	
	for image in pyramid:
		height, width, channels = np.shape(image);
		
		y_pad = (height%WINDOW_STEP)//2;
		x_pad = (width%WINDOW_STEP)//2;
		
		x_end = width-WINDOW_WIDTH-WINDOW_STEP;
		y_end = height-WINDOW_HEIGHT-WINDOW_STEP;
		
		for y in range(y_pad, y_end, WINDOW_STEP):
			for x in range(x_pad, x_end, WINDOW_STEP):
				window = image[y:y+WINDOW_HEIGHT, x:x+WINDOW_WIDTH];
				features = np.transpose(hog.compute(window));
				
				prediction = classifier.predict(features);
				if (prediction==1):
					windows.append(np.array([
											x*scale,
										 	y*scale,
										 	(x+WINDOW_WIDTH)*scale,
										 	(y+WINDOW_HEIGHT)*scale,
										 	classifier.decision_function(features)
										]));
		
		scale *= SCALE_STEP;
	
	
	end = time.time();
	elapsed = end - start;
	windows = non_max_suppression(np.array(windows), probs=None, overlapThresh=0.35)
	n_windows = len(windows);
	
	for i in range(n_windows):
		x1 = int(windows[i][0]);
		y1 = int(windows[i][1]);
		x2 = int(windows[i][2]);
		y2 = int(windows[i][3]);
		cv2.rectangle(orig_image, (x1, y1), (x2, y2), (255, 0, 0), 2);
	
	print("Detected %d windows in %.2f seconds"%(n_windows, elapsed));
	
	cv2.imshow("windows", orig_image);
	#cv2.waitKey(0);
	#cv2.destroyAllWindows();
	

def nonMaximumSuppression(windows, overlap_threshold):
	if not len(windows):
		return np.array([])
	# windows[:,0], windows[:,1] contain x,y of top left corner
	# windows[:,2], windows[:,3] contain x,y of bottom right corner
	I = np.argsort(windows[:,4])[::-1]
	area = (windows[:,2]-windows[:,0]+1) * (windows[:,3]-windows[:,1]+1)
	chosen = []

	while len(I):
		i = I[0]
		# Dims of intersections between window i and the rest
		width = np.maximum(0.0, np.minimum(windows[i,2], windows[I,2])-
			np.maximum(windows[i,0], windows[I,0])+1)
		height = np.maximum(0.0, np.minimum(windows[i,3], windows[I,3])-
			np.maximum(windows[i,1], windows[I,1])+1)
		overlap = (width*height).astype(np.float32)/area[I]
		mask = overlap<overlap_threshold
		I = I[mask]
		if mask.shape[0]-np.sum(mask) > 1 :
			chosen.append(i)
	return windows[chosen]

	
	
	
	
	
	
	
	
	
	
	
	
	
	
