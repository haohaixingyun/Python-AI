import tensorflow as tf


batch_size = 8

w1= tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w1= tf.Variable(tf.random_normal([3,1],stddev=1,seed=2))