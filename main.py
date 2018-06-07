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
from matplotlib import pyplot as plt
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
from matplotlib.ticker import MultipleLocator

warnings.simplefilter("ignore",
							 category = MatplotlibDeprecationWarning);
#Ignorar los warnings de matplotlib


ORIG_NEG_PATH = "database/INRIAPerson/Train/neg";

TRAINING_POS_PATH = "database/INRIAPerson/96X160H96/Train/pos_bigdata";
TRAINING_NEG_PATH = "database/INRIAPerson/96X160H96/Train/neg_bigdata";

TESTING_POS_PATH = "database/INRIAPerson/70X134H96/Test/pos";
TESTING_NEG_PATH = "database/INRIAPerson/70X134H96/Test/neg";

TRAINING_FEATURES_PATH = "extracted_features/training_features2.npy";
TRAINING_LABELS_PATH		= "extracted_features/training_labels2.npy";

TESTING_FEATURES_PATH 	= 	"extracted_features/testing_features2.npy";
TESTING_LABELS_PATH		=	"extracted_features/testing_labels2.npy";

CLASSIFIER_PATH = "classifiers/linear_soft.pkl";

kx = np.asarray([[-1], [0], [1]])
ky = np.asarray([[-1], [0], [1]])
DERIV_KERNEL = (kx,ky);

def classifyVideo():
	classifier = joblib.load(CLASSIFIER_PATH);
	
	root = tk.Tk()
	root.withdraw();
	
	video_path = filedialog.askopenfilename();
	
	cap = cv2.VideoCapture(video_path)
	
	
	count = 0
	while cap.isOpened():
		ret, frame = cap.read()
		
		if((count%25)==0):
			HumanDetection.detect(frame, classifier)
		
		count = count+1
		if cv2.waitKey(10)&0xFF==ord('q'):
			break
	


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
	
	#classifier = svm.LinearSVC(C=0.039062);
	classifier = svm.LinearSVC(C=6.666);
	classifier.fit(features,labels);
	
	joblib.dump(classifier, CLASSIFIER_PATH);

'''
def generateClassifier(features, labels):

	classifier = svm.LinearSVC(C=0.01);
	classifier.fit(features,labels);
	
	joblib.dump(classifier, CLASSIFIER_PATH);
'''

def generateClassifierWithRange():
	features_training = np.load(TRAINING_FEATURES_PATH);
	labels_training = np.load(TRAINING_LABELS_PATH);
	features_test = np.load(TESTING_FEATURES_PATH);
	labels_test = np.load(TESTING_LABELS_PATH);
	
	v_c = []
	
	for i in range(-12,5):
		v_c.append(2**i);
		
	 
	fpr = np.zeros((len(v_c)));
	tnr = np.zeros((len(v_c)));
	miss = np.zeros((len(v_c)));
	i = 0;
	
	for c in v_c:
		classifier = svm.LinearSVC(C=c);
		classifier.fit(features_training, labels_training);
		
		predictions = classifier.predict(features_test);
		n_positives = np.sum(predictions==labels_test);
		score = (n_positives/len(labels_test));
		
		
		
		confusion = metrics.confusion_matrix(labels_test, predictions);
		
		print(confusion)
		print(("%f : %f")%(c, score*100));
		
		fpr[i] = (confusion[0][1]/np.sum(confusion[0]));
		tnr[i] = (confusion[1][0]/np.sum(confusion[1]));
		miss[i] = 1-score;
		i+=1;
	
	plt.figure(1);
	plt.semilogx(v_c, 10*np.log10(fpr), basex=2, color="r", marker="s", linestyle="--", markerfacecolor="none",
				  linewidth=0.75, label="Falsos positivos");
	plt.hold(True);
	
	plt.semilogx(v_c, 10*np.log10(tnr), basex=2, color="b", marker="^", linestyle="--", markerfacecolor="none",
				  linewidth=0.75, label="Verdaderos negativos");
	
	plt.xlabel("Factor de penalizacion (C)");
	plt.ylabel("Error [dB]");
	plt.title("Tasas de errores frente a regularización");
	plt.grid(True, linestyle="--", linewidth=0.5);
	
	plt.legend();
	
	
	plt.figure(2);
	plt.semilogx(v_c, 10*np.log(miss), basex=2, color="k", marker="o", linestyle="--", markerfacecolor="none",
				  linewidth=0.75, label="Tasa de error");
	
	plt.xlabel("Factor de penalizacion (C)");
	plt.ylabel("Error [dB]");
	plt.title("Tasa de error frente a regularización");
	plt.grid(True, linestyle="--", linewidth=0.5);
	
	plt.show();
	
	

def test():
	features = np.load(TESTING_FEATURES_PATH);
	labels = np.load(TESTING_LABELS_PATH);
	classifier = joblib.load(CLASSIFIER_PATH);
	
	predictions = classifier.predict(features);
	n_positives = np.sum(predictions==labels);
	accuracy = (n_positives/len(labels))*100;
	
	confusion_matrix = metrics.confusion_matrix(labels,predictions);
	
	print("Accuracy : %f"%(accuracy));
	print("Confussion matrix : ");
	print(confusion_matrix);

def test2():
	features = np.load(TESTING_FEATURES_PATH);
	labels = np.load(TESTING_LABELS_PATH);
	
	hog = cv2.HOGDescriptor()
	parameters = cv2.HOGDescriptor_getDefaultPeopleDetector();
	
	classifier = svm.LinearSVC(C=parameters[-1]);
	classifier.coef_ = np.transpose(parameters[0:3780]);
	
	#classifier.set_params(coef_=np.transpose(parameters[0:3780]));
	
	predictions = classifier.predict(features);
	
	n_positives = np.sum(predictions==labels);
	
	accuracy = (n_positives/len(labels))*100;
	
	print(accuracy);

def main():
	#generateClassifier();
	#generateTestingFeatures();
	#generateTrainingFeatures();
	#test();
	generateClassifierWithRange()
	
	#classifyImage();
	#classifyVideo();
	
	#generateTrainingFeatures();
	#generateClassifier();
	#test();
	
	#test2();
	
	

















































































































if(__name__=="__main__"):
	main();

