import tensorflow as tf
import numpy as np
from CNN import get_cnn
import os
import time
import datetime
import pickle
import csv
import gensim

K = 400 # word2vec embedding dimension
L = 50 # 50 words limit for each sentence
NUM_CLASSES = 2

_x = tf.placeholder(tf.float32, [None, L, K], name='input_x')
_y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='input_y')

class Predictor:
	
	def __init__(self, W2V_MODEL_PATH, CNN_MODEL_PATH):
		self.scores, self.predictions, self.loss, self.accuracy = get_cnn(_x, _y, 0)
		print('CNN architecture created.')
		self.w2v = gensim.models.KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=True, limit=3000000)
		print('W2V model loaded.')
		self.sess = tf.Session(config=tf.ConfigProto(
			device_count = {'GPU': 0}
		))
		saver = tf.train.Saver()
		saver.restore(self.sess, CNN_MODEL_PATH)
		print('CNN parameteres restored.')
	
	def make_prediction(self, tweet_text):
		tmp = []
		for word in tweet_text.split(' '):
			try:
				tmp.append(self.w2v[word]) # find word from W2V
			except:
				''' OOV word '''
				tmp.append(np.random.uniform(-0.5,0.5,K)) # random vector range:(-0.5, 0.5)
			if len(tmp) == L:
				break # 50 words limit!!
		while len(tmp) < L:
			''' pad with zero vectors if less than 50 words '''
			tmp.append(np.zeros(K)) 
		tmp = np.array(tmp)
		pred = self.sess.run([self.predictions], feed_dict={_x: [tmp]})
		return pred[0][0]