import numpy as np
import cv2
import os
from sklearn import svm, linear_model, metrics
from database_creation import *
from features_extraction import *
import human_detection as HumanDetection
from sklearn.externals import joblib
import tkinter as tk
from tkinter import filedialog
#import h5py

ORIG_NEG_PATH = "database/INRIAPerson/Train/neg";

TRAINING_POS_PATH = "database/INRIAPerson/96X160H96/Train/pos_bigdata";
TRAINING_NEG_PATH = "database/INRIAPerson/96X160H96/Train/neg_bigdata";

TESTING_POS_PATH = "database/INRIAPerson/70X134H96/Test/pos";
TESTING_NEG_PATH = "database/INRIAPerson/70X134H96/Test/neg";

TRAINING_FEATURES_PATH = "extracted_features/training_features_bigdata_100.npy";
TRAINING_LABELS_PATH		= "extracted_features/training_labels_bigdata_100.npy";

TESTING_FEATURES_PATH 	= 	"extracted_features/testing_features2.npy";
TESTING_LABELS_PATH		=	"extracted_features/testing_labels2.npy";

CLASSIFIER_PATH = "classifiers/linear3.pkl";

kx = np.asarray([[-1], [0], [1]])
ky = np.asarray([[-1], [0], [1]])
DERIV_KERNEL = (kx,ky);

def classifyVideo():
	classifier = joblib.load(CLASSIFIER_PATH);
	
	root = tk.Tk()
	root.withdraw();
	
	#while(True):
	#	image_path = filedialog.askopenfilename();
	#	HumanDetection.detect(image_path,classifier);
	
	image_path = filedialog.askopenfilename();
	
	cap = cv2.VideoCapture(image_path)
	
	
	count = 0
	while cap.isOpened():
		ret, frame = cap.read()
		
		if((count%25)==0):
			HumanDetection.detect(frame, classifier)
		
		count = count+1
		if cv2.waitKey(10)&0xFF==ord('q'):
			break
	
	'''
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector());
	while True:
		_, frame = cap.read()
		found, w = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)
		
		for x1, y1, w, h in found:
			x2 = x1 + w;
			y2 = y1 + h;
			cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2);
		
		cv2.imshow('feed', frame)
		ch = 0xFF&cv2.waitKey(1)
		if ch==27:
			break
	cv2.destroyAllWindows()
	'''
	


def classifyImage():
	classifier = joblib.load(CLASSIFIER_PATH);
	
	root = tk.Tk()
	root.withdraw();
	
	while(True):
		image_path = filedialog.askopenfilename();
		image = cv2.imread(image_path);
		HumanDetection.detect(image,classifier);
	

def generateTrainingFeatures():
	features, labels = obtainDataFeatures(TRAINING_POS_PATH,
													  TRAINING_NEG_PATH);
	
	#np.save(TRAINING_FEATURES_PATH, features);
	#np.save(TRAINING_LABELS_PATH, labels);
	
	generateClassifier(features, labels);
	
	h5_file = h5py.File(TRAINING_FEATURES_PATH, 'w');
	h5_file.create_dataset("features", data=features);
	
	h5_file.close();
	
def generateTestingFeatures():
	features, labels = obtainDataFeatures(TESTING_POS_PATH,
													  TESTING_NEG_PATH);
	
	np.save(TESTING_FEATURES_PATH, features);
	np.save(TESTING_LABELS_PATH, labels);

def generateClassifier():
	features = np.load(TRAINING_FEATURES_PATH);
	labels = np.load(TRAINING_LABELS_PATH);
	
	classifier = svm.LinearSVC(C=0.01);
	classifier.fit(features,labels);
	
	joblib.dump(classifier, CLASSIFIER_PATH);

def generateClassifier(features, labels):

	classifier = svm.LinearSVC(C=0.01);
	classifier.fit(features,labels);
	
	joblib.dump(classifier, CLASSIFIER_PATH);
	
def generateClassifierWithRange():
	features_training = np.load(TRAINING_FEATURES_PATH);
	labels_training = np.load(TRAINING_LABELS_PATH);
	features_test = np.load(TESTING_FEATURES_PATH);
	labels_test = np.load(TESTING_LABELS_PATH);
	
	C = []
	
	for i in range(0,7):
		C.append(10**(-i));
	
	for c in C:
		classifier = svm.LinearSVC(C=c);
		classifier.fit(features_training, labels_training);
		score = classifier.score(features_test, labels_test);
		print(score);
	

def test():
	features = np.load(TESTING_FEATURES_PATH);
	labels = np.load(TESTING_LABELS_PATH);
	classifier = joblib.load(CLASSIFIER_PATH);
	
	predictions = classifier.predict(features);
	
	n_positives = np.sum(predictions==labels);
	
	accuracy = (n_positives/len(labels))*100;
	
	print(accuracy);

def main():
	#generateClassifier();
	#generateTestingFeatures();
	#generateTrainingFeatures();
	#test();
	#generateClassifierWithRange()
	
	
	classifyImage();
	#classifyVideo();
	
	#generateTrainingFeatures();
	#generateClassifier();
	#test();
	
	'''
	features, labels = obtainDataFeatures(TRAINING_POS_PATH,
													  TRAINING_NEG_PATH);
	
	classifier = svm.LinearSVC(C=0.01);
	classifier.fit(features,labels);
	
	joblib.dump(classifier, "classifiers/linear_bigdata_50.pkl");
	
	test();
													  
	h5_file = h5py.File("extracted_features/training_features_bigdata_50.h5", 'w');
	h5_file.create_dataset("features", data=features);
	
	h5_file.close();
	'''

















































































































if(__name__=="__main__"):
	main();

