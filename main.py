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

ORIG_NEG_PATH = "database/INRIAPerson/Train/neg";

TRAINING_POS_PATH = "database/INRIAPerson/96X160H96/Train/pos";
TRAINING_NEG_PATH = "database/INRIAPerson/96X160H96/Train/neg";

TESTING_POS_PATH = "database/INRIAPerson/70X134H96/Test/pos";
TESTING_NEG_PATH = "database/INRIAPerson/70X134H96/Test/neg";

TRAINING_FEATURES_PATH = "extracted_features/training_features2.npy";
TRAINING_LABELS_PATH		= "extracted_features/training_labels2.npy";

TESTING_FEATURES_PATH 	= 	"extracted_features/testing_features2.npy";
TESTING_LABELS_PATH		=	"extracted_features/testing_labels2.npy";

CLASSIFIER_PATH = "classifiers/linear2.pkl";

kx = np.asarray([[-1], [0], [1]])
ky = np.asarray([[-1], [0], [1]])
DERIV_KERNEL = (kx,ky);

def classifyImage():
	classifier = joblib.load(CLASSIFIER_PATH);
	
	root = tk.Tk()
	root.withdraw();
	
	while(True):
		image_path = filedialog.askopenfilename();
		HumanDetection.detect(image_path,classifier);
	

def generateTrainingFeatures():
	features, labels = obtainDataFeatures(TRAINING_POS_PATH,
													  TRAINING_NEG_PATH);
	
	np.save(TRAINING_FEATURES_PATH, features);
	np.save(TRAINING_LABELS_PATH, labels);

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

def test():
	features = np.load(TESTING_FEATURES_PATH);
	labels = np.load(TESTING_LABELS_PATH);
	classifier = joblib.load(CLASSIFIER_PATH);
	
	predictions = classifier.predict(features);
	
	n_positives = np.sum(predictions==labels);
	
	accuracy = (n_positives/len(labels))*100;
	
	print(accuracy);


def main():
	#trainModel();
	#generateClassifier();
	#generateTestingFeatures();
	#test();
	classifyImage();




















































































































if(__name__=="__main__"):
	main();