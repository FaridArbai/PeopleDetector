import numpy as np
import tensorflow as tf
from main import *
import h5py;

'''
	The following neural network has an architecture of 3 fully connected
	layers of 1k neurons each followed by a soft max cross-entropy one
	which decides if the current descriptor represents a person or if not.
	
	Note that this neural network was made for research purposes, hence
	the SVM model clearly outperforms the current model. Training with 100k
	samples gives 82.98% accuracy with this DNN versus the 99.85% accomplished
	with the linear SVM.
'''

def trainNeuralNetwork():
	file = h5py.File("extracted_features/training_features_bigdata_50.h5", 'r');
	features_training = file["features"].value;
	file.close();
	labels_training = np.load("extracted_features/training_labels_bigdata_50.npy");
	features_testing = np.load(TESTING_FEATURES_PATH);
	labels_testing = np.load(TESTING_LABELS_PATH);
	
	labels_testing = np.transpose([labels_testing, 1-labels_testing]);
	
	print(features_training.shape);
	
	n_train_data = len(labels_training);
	
	FEATURE_SIZE = 3780;
	x = tf.placeholder(tf.float32, [None, FEATURE_SIZE])
	
	Nl1 = 1000 
	W1 = tf.Variable(tf.zeros([FEATURE_SIZE, Nl1]))
	b1 = tf.Variable(tf.zeros([Nl1]))
	y1 = tf.nn.sigmoid(tf.matmul(x, W1)+b1)
	keep_prob = tf.placeholder(tf.float32)
	y1_drop = tf.nn.dropout(y1, keep_prob, seed=2)
	
	W2 = tf.Variable(tf.zeros([Nl1, Nl1]))
	b2 = tf.Variable(tf.zeros([Nl1]))
	y2 = tf.nn.sigmoid(tf.matmul(y1_drop, W2)+b2)
	y2_drop = tf.nn.dropout(y2, keep_prob, seed=2)
	
	W23 = tf.Variable(tf.zeros([Nl1, Nl1]))
	b23 = tf.Variable(tf.zeros([Nl1]))
	y23 = tf.nn.sigmoid(tf.matmul(y2_drop, W23)+b23)
	y23_drop = tf.nn.dropout(y23, keep_prob, seed=2)
	
	W3 = tf.Variable(tf.zeros([Nl1, 2]))
	b3 = tf.Variable(tf.zeros([2]))
	y3 = tf.matmul(y23_drop, W3)+b3 
	
	y = tf.nn.softmax(y3)
	
	y_ = tf.placeholder(tf.float32, [None, 2]) # These are the actual outputs
	
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	
	batch_size = 1; #Just for testing purposes
	
	for i in range(n_train_data):
		batch_xs = features_training[i:i+batch_size, :];
		is_person = labels_training[i:i+batch_size];
		batch_ys = np.transpose([is_person, (1-is_person)]);
		print((i/n_train_data)*100);
		sess.run(train_step, feed_dict={
			x:batch_xs, y_:batch_ys, keep_prob:0.75})
	
	Acc = sess.run(accuracy, feed_dict={x:features_testing, y_:labels_testing, keep_prob:1.0})
	
	print('Testing results of '+str(100.*Acc)+' %')	
	
trainNeuralNetwork();