import numpy as np
import h5py
import cv2
import os

WINDOW_HEIGHT = 128;
WINDOW_WIDTH = 64;

def readDatabaseImage(path):
	image = cv2.imread(path);
	height, width, channels = np.shape(image);
	
	#Las imagenes pueden tener un pequeño padding lateral o
	#vertical, por lo que se elimina durante su lectura
	
	x_pad = (width - WINDOW_WIDTH)//2;
	y_pad = (height - WINDOW_HEIGHT)//2;
	
	image = image[y_pad:y_pad+WINDOW_HEIGHT, x_pad:x_pad+WINDOW_WIDTH];
	
	return image;


winSize 				= (64,128);
'''
	Tamaño de las imagenes de las que
	vamos a extraer caracteristicas
'''

nbins 				= 9;
'''
	Numero de bins que contiene cada
	histograma de gradientes, desde
	0 grados hasta 180 grados en este
	caso de 20 grados de separación.
'''

cellSize 			= (8,8);
'''
	Tamaño de la celula para la cual
	sacar el histograma de 9 bins
'''

blockSize 			= (16,16);
'''
	Tamaño del bloque de celulas
	que se va a usar para normalizar
	Como es 2x2 celulas, queda en
	16x16 px.
'''
blockStride			= (8,8);
'''
	Desplazamiento de los bloques, al
	solaparse a 50% es de 1 bloque y
	por lo tanto 8px
'''

histogramNormType	= 0;
'''
	Código para la norma L2-Hys
'''

L2HysThreshold 	= 0.2;
'''
	Valor de clipping para la saturación
	o histéresis de L2-Hys.
'''

EXTRACTOR = cv2.HOGDescriptor(winSize, blockSize,
										blockStride, cellSize,
										nbins, histogramNormType,
										L2HysThreshold);


def generateFeatures(pos_path, neg_path, features_dst, labels_dst):
	'''
	Esta funcion genera las 3780 caracteristicas de cada imagen
	que se encuentra en los directorios pos_path y neg_path
	y almacena las caracteristicas en un fichero .h5 en
	features_dst y las etiquetas en un fichero .npy dentro
	de labels_dst
	
	:param pos_path:
	:param neg_path:
	:param features_dst:
	:param labels_dst:
	:return:
	'''
	
	pos_filenames = os.listdir(pos_path)
	neg_filenames = os.listdir(neg_path)
	
	n_pos = len(pos_filenames);
	n_neg = len(neg_filenames);
	n_data = n_pos + n_neg;
	
	features = np.zeros((n_pos+n_neg, 3780));
	
	classes_code = np.array([0,1]);
	
	labels = np.concatenate((np.repeat(1,n_pos),
									  np.repeat(0,n_neg)));
	
	count = 0;
	filenames = pos_filenames+neg_filenames;
	
	'''
		Como sabemos cuantas imagenes positivas hay, en el
		momento en que count >= n_pos filenames solo contedra
		imagenes negativas.
	'''
	
	for filename in filenames:
		if (count<n_pos):
			folder = pos_path;
		else:
			folder = neg_path;
		
		image_path = ("%s/%s")%(folder, filename);
		image = readDatabaseImage(image_path);
		
		feature = EXTRACTOR.compute(image);
		features[count, :] = np.reshape(feature, (3780,));
		
		count += 1;
	
	#Guardo las etiquetas en un fichero .npy
	np.save(labels_dst, labels);
	
	#Guardo las características en un fichero .h5
	h5_file = h5py.File(features_dst, 'w');
	h5_file.create_dataset("features", data=features);
	h5_file.close();


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	