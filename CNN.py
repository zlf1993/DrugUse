'''
Title: Convolutional neural network model for natural language processing
==========================================================================
Model architecture:
	input size: [?, L=50, K=400] # L: length of sentence; K: dimension of embedding
'''

import tensorflow as tf

#LOGDIR = 'log_tensorboard/'

K = 400 # word2vec embedding dimension
L = 50 # 50 words limit for each sentence
FILTER_SIZES = [3,4,5,6,7] # the height of filters
NUM_FILTERS = 64
NUM_CLASSES = 2

# Conv-layer
def conv_layer(input):
	pooled_outputs = [] 
	for i, filter_size in enumerate(FILTER_SIZES): 
		with tf.name_scope('conv-maxpool-'+str(filter_size)):
			filter_shape = [filter_size, K, 1, NUM_FILTERS] # [height, width, in-channels, out-channels]
			w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
			b = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]), name="b")
			''' Convolution '''
			conv_layer = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='VALID', name='conv')
			''' ReLU activation function '''
			relu = tf.nn.relu(tf.nn.bias_add(conv_layer, b), name='relu')
			''' Maxpooling '''
			pool = tf.nn.max_pool(
				relu, 
				ksize=[1, L-filter_size+1, 1, 1],
				strides=[1,1,1,1],
				padding='VALID',
				name='pool'
			)
			pooled_outputs.append(pool)

	''' Combine all the pooled outputs '''
	total_num_filters = NUM_FILTERS*len(FILTER_SIZES)
	cat_pool = tf.concat(pooled_outputs, 3)
	pool_flat = tf.reshape(cat_pool, [-1, total_num_filters])

	return pool_flat, total_num_filters

# Get neural network model
def get_cnn(x, Y, pos_weight, get_deep_feature = False):
	X = tf.reshape(x, [-1, L, K, 1])

	''' Conv-ReLU-maxpool layer '''
	pool_flat, total_num_filters = conv_layer(X)

	''' Output layer '''
	with tf.name_scope('output-layer'):
		w = tf.get_variable('w', shape=[total_num_filters, NUM_CLASSES],
			initializer=tf.contrib.layers.xavier_initializer())
		b = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b")
		scores = tf.nn.xw_plus_b(pool_flat, w, b, name='scores')
		predictions = tf.argmax(scores, 1, name='predictions')

	''' Softmax cross-entropy loss '''
	with tf.name_scope('loss'):
		#batch_losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=Y)
		batch_losses = tf.nn.weighted_cross_entropy_with_logits(logits=scores, targets=Y, pos_weight = pos_weight)
		loss = tf.reduce_mean(batch_losses)

	''' Accuracy '''
	with tf.name_scope('accuracy'):
		correct_predictions = tf.equal(predictions, tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
	if get_deep_feature:
		return scores, predictions, loss, accuracy, pool_flat
	else:
		return scores, predictions, loss, accuracy