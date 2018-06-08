import numpy as np
import cv2
import os
from sklearn import svm, linear_model, metrics
from database_creation import *
from features_extraction import *
import time
import imutils.object_detection

EXTRACTOR = cv2.HOGDescriptor();

def showWindows(orig_image, classifier):
	'''
	Esta función detecta a las personas de la imagen en varias
	escalas usando el clasificador SVM proporcionado y usando
	NMS para suprimir los solapamientos. Una vez detectados
	todos los rectángulos, muestra estos junto a la imagen a
	la par que los recorta a parte por si se quieren analizar.
	
	:param orig_image: Imagen original en la que hacer la
		detección
	:param classifier: Clasificador SVM ya entrenado
	'''
	
	WINDOW_HEIGHT = 128;
	WINDOW_WIDTH = 64;
	SCALE_STEP = 1.2;
	WINDOW_STEP = 8;
	
	GAUSS_KERNEL_SIZE = (5,5);
	GAUSS_SIGMA = 0.6;
	
	height, width, channels = np.shape(orig_image);
	
	pyramid = [orig_image]

	new_width = width//SCALE_STEP;
	new_height = height//SCALE_STEP;
	
	while (new_height>WINDOW_HEIGHT and new_width>WINDOW_WIDTH):
		blurred = cv2.GaussianBlur(pyramid[-1], ksize=GAUSS_KERNEL_SIZE,
											sigmaX=GAUSS_SIGMA, sigmaY=GAUSS_SIGMA);
		pyramid.append(cv2.resize(src=blurred, dsize=(int(new_width),int(new_height))))
		
		new_height //= SCALE_STEP;
		new_width //= SCALE_STEP;
	
	
	scaling_factor = 1;
	windows = [];
	
	for image in pyramid:
		height, width, channels = np.shape(image);
		
		y_pad = (height%WINDOW_STEP)//2;
		x_pad = (width%WINDOW_STEP)//2;
		
		x_end = width-WINDOW_WIDTH-WINDOW_STEP;
		y_end = height-WINDOW_HEIGHT-WINDOW_STEP;
		
		for y in range(y_pad, y_end, WINDOW_STEP):
			for x in range(x_pad, x_end, WINDOW_STEP):
				window = image[y:y+WINDOW_HEIGHT, x:x+WINDOW_WIDTH];
				features = np.transpose(EXTRACTOR.compute(window));
				
				prediction = classifier.predict(features);
				if (prediction==1):
					windows.append(np.array([
											x*scaling_factor,
										 	y*scaling_factor,
										 	(x+WINDOW_WIDTH)*scaling_factor,
										 	(y+WINDOW_HEIGHT)*scaling_factor,
										 	classifier.decision_function(features)
										]));
		
		scaling_factor *= SCALE_STEP;
	

	'''
		Supresión de no-máximos usando la función non_max_suppression del
		paquete imutils, el estándar para hacer NMS.
	'''
	windows = imutils.object_detection.non_max_suppression(np.array(windows),
																			 probs=None,
																			 overlapThresh=0.65)
	n_windows = len(windows);
	
	cp_image = orig_image.copy();
	
	for i in range(n_windows):
		x1 = int(windows[i][0]);
		y1 = int(windows[i][1]);
		x2 = int(windows[i][2]);
		y2 = int(windows[i][3]);
		cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 0, 255), 2);
		cv2.imshow("Cropped window number %d"%(i+1), cp_image[y1:y2,x1:x2]);
	
	cv2.imshow("Detected windows",orig_image);
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	