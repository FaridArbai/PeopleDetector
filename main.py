import numpy as np
import cv2
import os
from sklearn import svm, linear_model, metrics
from database_creation import *
from features_extraction import *
import human_detection as HumanDetection
from sklearn.externals import joblib

ORIG_NEG_PATH = "database/INRIAPerson/Train/neg";

TRAINING_POS_PATH = "database/INRIAPerson/96X160H96/Train/pos";
TRAINING_NEG_PATH = "database/INRIAPerson/96X160H96/Train/neg";

TESTING_POS_PATH = "database/INRIAPerson/70X134H96/Test/pos";
TESTING_NEG_PATH = "database/INRIAPerson/70X134H96/Test/neg";

TRAINING_FEATURES_PATH = "extracted_features/training_features.npy";
TRAINING_LABELS_PATH		= "extracted_features/training_labels.npy";

TESTING_FEATURES_PATH 	= 	"extracted_features/testing_features.npy";
TESTING_LABELS_PATH		=	"extracted_features/testing_labels.npy";

CLASSIFIER_PATH = "classifiers/linear.pkl";

kx = np.asarray([[-1], [0], [1]])
ky = np.asarray([[-1], [0], [1]])
DERIV_KERNEL = (kx,ky);

def main():
	classifier = joblib.load(CLASSIFIER_PATH);
	HumanDetection.detect("testing_pedestrians/person.png",classifier);
	
	
	
	'''
		#this is to generate and save the svm
		training_features = np.load(TRAINING_FEATURES_PATH);
		training_labels = np.load(TRAINING_LABELS_PATH);

		testing_features = np.load(TESTING_FEATURES_PATH);
		testing_labels = np.load(TESTING_LABELS_PATH);


		classifier = svm.LinearSVC(C=0.01);

		classifier.fit(training_features, training_labels);

		joblib.dump(classifier,CLASSIFIER_PATH);
	'''
	






















































































































if(__name__=="__main__"):
	main();