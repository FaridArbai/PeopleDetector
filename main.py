import cv2
import database_creation as DatabaseCreation
#import features_extraction as FeaturesExtraction
import machine_learning as MachineLearning
import people_detection as PeopleDetection
import visualization as Visualization
from sklearn.externals import joblib
import tkinter as tk
from tkinter import filedialog

TRAINING_POS_PATH = "database/INRIAPerson/96X160H96/Train/pos_bigdata";
TRAINING_NEG_PATH = "database/INRIAPerson/96X160H96/Train/neg_bigdata";

TESTING_POS_PATH = "database/INRIAPerson/70X134H96/Test/pos";
TESTING_NEG_PATH = "database/INRIAPerson/70X134H96/Test/neg";

TRAINING_FEATURES_PATH = "extracted_features/training_features2.npy";
TRAINING_LABELS_PATH		= "extracted_features/training_labels2.npy";

TESTING_FEATURES_PATH 	= 	"extracted_features/testing_features2.npy";
TESTING_LABELS_PATH		=	"extracted_features/testing_labels2.npy";

CLASSIFIER_PATH = "classifiers/clasificador_final.pkl";

		
def detectPeople(classifier):
	image_path = filedialog.askopenfilename();
	image = cv2.imread(image_path);
	PeopleDetection.showWindows(image, classifier);
	#visualizeHog(image);

def plotHog():
	image_path = filedialog.askopenfilename();
	image = cv2.imread(image_path);
	Visualization.visualizeHog(image);
	
def main():
	#1. Cargar el clasificador
	classifier = joblib.load(CLASSIFIER_PATH);
	
	#2. Generar un boton para escoger imagenes
	root = tk.Tk();
	root.title("PeopleDetector");
	root.minsize(128, 128);
	
	detect_button = tk.Button(master=root,
							 text="Detectar Personas",
							 command=lambda:detectPeople(classifier),
							 );
	
	hog_button = tk.Button(master=root,
									  text="Mostrar HOG",
									  command=plotHog,
									  );
	
	detect_button.place(relx=0.5,rely=0.33, anchor=tk.CENTER);
	hog_button.place(relx=0.5, rely=0.66, anchor=tk.CENTER);
	
	#3. Entrar en un bucle en el que el usuario
	# visualiza imagenes
	root.mainloop();
	
	

















































































































if(__name__=="__main__"):
	main();

