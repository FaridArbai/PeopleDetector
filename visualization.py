from sklearn import svm, metrics
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog as plothog
from skimage import exposure
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning

def visualizeHog(orig_image):
	'''
	Esta función genera un par de imagenes representando la imagen original y la
	distribución espacial de su gistograma de gradientes orientados.
	
	:param orig_image: Imagen original en formato BGR (OpenCV)
	'''
	
	image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY);
	fd, hog_image = plothog(image, orientations=9, pixels_per_cell=(8, 8),
									cells_per_block=(2, 2), visualise=True, block_norm="L2-Hys")
	
	fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
	
	ax1.axis('off')
	ax1.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)
	ax1.set_title('Input image')
	
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
	
	ax2.axis('off')
	ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
	ax2.set_title('Histogram of Oriented Gradients')
	plt.show()
	
	cv2.imshow("HOG", hog_image_rescaled);


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
	
	warnings.simplefilter("ignore",
								 category=MatplotlibDeprecationWarning);
	#Ignorar los warnings de matplotlib
	
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