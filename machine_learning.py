from sklearn import svm, metrics
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt

def generateOptimalClassifier(training_features_path,
										training_labels_path,
										testing_features_path,
										testing_labels_path,
										classifier_dst_path):
	'''
	Esta función entrena N clasificadores SVM lineales con
	distintos valores de factor de penalizacion, en el rango
	2**(-15) hasta 2**4 con pasos multiplicativos de 2. Una
	vez se encuentra el clasificador optimo, se guarda en
	el archivo classifier_dst_path en formato .pkl para
	cargarlo de cara a futuras clasificaciones y no tener
	que entrenar en cada ejecución.
	
	:param training_features_path: Ruta donde se encuentra
	un fichero .h5 con las características extraidas de la
	base de datos de entrenamiento.
	
	:param training_labels_path: Ruta donde se encuentra
	un fichero .npy con las etiquetas asociadas a cada
	vector de características de la base de datos de
	entrenamiento
	
	:param testing_features_path: Ruta donde se encuentra
	un fichero .npy con las características extraídas de la
	base de datos de testing.
	
	:param testing_labels_path: Ruta donde se encuentra
	un fichero .npy con las etiquetas asociadas a los
	vectores de características de la base de datos
	de testing.
	'''
	features_training = np.load(training_features_path);
	labels_training = np.load(training_labels_path);
	features_test = np.load(testing_features_path);
	labels_test = np.load(testing_labels_path);
	
	v_c = []; # vector con los valores de C
	
	for i in range(-12, 5):
		v_c.append(2**i); # valores de C
	
	optimal_classifier = [];
	miss_opt = 1;
	
	for c in v_c:
		classifier = svm.LinearSVC(C=c);
		classifier.fit(features_training, labels_training);
		
		predictions = classifier.predict(features_test);
		n_positives = np.sum(predictions==labels_test);
		
		score = (n_positives/len(labels_test));
		miss = 1-score;
		
		if(miss < miss_opt):
			miss_opt = miss;
			optimal_classifier = classifier;
	
	joblib.dump(classifier_dst_path, optimal_classifier);


def generateClassifierGraphs(training_features_path,
										training_labels_path,
										testing_features_path,
										testing_labels_path):
	'''
	Esta función entrena N clasificadores SVM lineales con
	distintos valores de factor de penalizacion, en el rango
	2**(-15) hasta 2**4 con pasos multiplicativos de 2 para
	sacar por cada uno métricas y gráficas de error para
	añadir al informe.
	
	:param training_features_path: Ruta donde se encuentra
	un fichero .h5 con las características extraidas de la
	base de datos de entrenamiento.
	
	:param training_labels_path: Ruta donde se encuentra
	un fichero .npy con las etiquetas asociadas a cada
	vector de características de la base de datos de
	entrenamiento
	
	:param testing_features_path: Ruta donde se encuentra
	un fichero .npy con las características extraídas de la
	base de datos de testing.
	
	:param testing_labels_path: Ruta donde se encuentra
	un fichero .npy con las etiquetas asociadas a los
	vectores de características de la base de datos
	de testing.
	'''
	features_training = np.load(training_features_path);
	labels_training = np.load(training_labels_path);
	features_test = np.load(testing_features_path);
	labels_test = np.load(testing_labels_path);
	
	v_c = []
	
	for i in range(-12, 5):
		v_c.append(2**i);
	
	fpr = np.zeros((len(v_c)));
	tnr = np.zeros((len(v_c)));
	miss = np.zeros((len(v_c)));
	i = 0;
	
	optimal_classifier = [];
	miss_opt = 1;
	
	for c in v_c:
		classifier = svm.LinearSVC(C=c);
		classifier.fit(features_training, labels_training);
		
		predictions = classifier.predict(features_test);
		n_positives = np.sum(predictions==labels_test);
		score = (n_positives/len(labels_test));
		
		miss = 1-score;
		confusion = metrics.confusion_matrix(labels_test, predictions);
		
		fpr[i] = (confusion[0][1]/np.sum(confusion[0]));
		tnr[i] = (confusion[1][0]/np.sum(confusion[1]));
		miss[i] = miss;
		i += 1;
	
	plt.figure(1);
	plt.semilogx(v_c, 10*np.log10(fpr), basex=2, color="r", marker="s",
					 linestyle="--", markerfacecolor="none",
					 linewidth=0.75, label="Falsos positivos");
	plt.hold(True);
	plt.semilogx(v_c, 10*np.log10(tnr), basex=2, color="b", marker="^",
					 linestyle="--", markerfacecolor="none",
					 linewidth=0.75, label="Verdaderos negativos");
	plt.xlabel("Factor de penalizacion (C)");
	plt.ylabel("Error [dB]");
	plt.title("Tasas de errores frente a regularización");
	plt.grid(True, linestyle="--", linewidth=0.5);
	plt.legend();
	
	plt.figure(2);
	plt.semilogx(v_c, 10*np.log(miss), basex=2, color="k", marker="o",
					 linestyle="--", markerfacecolor="none",
					 linewidth=0.75, label="Tasa de error");
	plt.xlabel("Factor de penalizacion (C)");
	plt.ylabel("Error [dB]");
	plt.title("Tasa de error frente a regularización");
	plt.grid(True, linestyle="--", linewidth=0.5);
	
	plt.show();
	
	