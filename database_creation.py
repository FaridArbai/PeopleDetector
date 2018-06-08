import numpy as np
import cv2
import os
from sklearn import svm, linear_model, metrics
import random
from features_extraction import *

#Paso 1: creacion de la base de datos

N_NEG_CROPS = 50;

def generateNegativeDatabase(orig_path, dst_path):
	'''
	Esta función genera la base de datos de muestras
	negativas recortando N_NEG_CROPS imagenes de
	64x128 px. obtenidas de orig_path y almacenando
	cada crop en dst_path
	
	:param orig_path:	 Ruta de la carpeta con las
	imagenes de las que extraer negativos.
	
	:param dst_path: Ruta de la carpeta en la
	que guardar cada imagen de 64x128 px.
	'''
	
	filenames = os.listdir(orig_path);
	
	# Para cada fichero se generan N_NEG_CROPS imagenes
	# en una posicion (x0,y0) totalmente aleatoria
	for name in filenames:
		orig_image_path = "%s/%s"%(orig_path, name);
		image = cv2.imread(orig_image_path);
		x0_max = image.shape[1]-64;
		y0_max = image.shape[0]-128;
	
		for i in range(N_NEG_CROPS):
				dst_image_path = "%d_%s/%s"%(i,dst_path, name);
				
				x0 = random.randint(0,x0_max);
				y0 = random.randint(0,y0_max);
				xf = x0 + 64;
				yf = y0 + 128;
				
				random_crop = image[x0:xf, y0:yf];
				
				cv2.imwrite(dst_image_path, random_crop);
	
	
	
	
	
	
	
	
	
	
	
	
	
	