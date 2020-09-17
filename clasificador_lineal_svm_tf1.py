import os
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.disable_v2_behavior()
session = tf.Session()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plotear = True
################ DATOS ##
#########################
from datasetMocks import linearSVM # modulo propio

all_values = np.array([ row for row in linearSVM ], dtype=np.float32)
x_y_all_values = np.array([ [x, y] for x,y,_z in all_values ], dtype=np.float32)
z_all_values = np.array([ z for _x,_y,z in all_values ], dtype=np.float32)

dataLen = len(x_y_all_values)
_x_y_rows, x_y_columns = x_y_all_values.shape
batch_size = 70

# ######## THE MODEL TRAINING BEGINS #
# ####################################
puntosXY = tf.placeholder(shape=[None, x_y_columns], dtype=tf.float32)
imagenZ = tf.placeholder(shape=[None], dtype=tf.float32)
pendiente = tf.Variable(tf.random_normal(shape=[x_y_columns, 1]))
ordenadaAlOrigen = tf.Variable(tf.random_normal(shape=[1]))

output = tf.reshape(tf.subtract(tf.matmul(puntosXY, pendiente), ordenadaAlOrigen), [-1])

regularization_loss = tf.divide(tf.square(tf.norm(pendiente, ord=2)), 2)
hinge_loss = tf.reduce_sum(tf.maximum(0., tf.subtract(1., tf.multiply(imagenZ, output))))
svm_loss = tf.add(regularization_loss, tf.multiply(1., hinge_loss))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(svm_loss)

init = tf.global_variables_initializer()
session.run(init)

for _ in range(100):
	rand_idx = np.random.choice(dataLen, size=batch_size)
	rand_xy = x_y_all_values[rand_idx]
	rand_z = np.transpose(z_all_values[rand_idx])
	session.run(train_step, feed_dict={puntosXY: rand_xy, imagenZ: rand_z})

# ######## TESTEO ##
# ##################
pend = session.run(pendiente)
ordOrigen = session.run(ordenadaAlOrigen)[0]
calculoSignado = lambda xy: np.sign(np.sum(np.subtract(np.matmul(xy, pend), ordOrigen))) # xy=[x, y], pend=[[w], [w]] 

predicciones = np.transpose(np.array([ calculoSignado(xy) for xy in x_y_all_values ], dtype=np.float32)).flatten()
predYtarget = np.c_[predicciones, z_all_values]
precisionResults = np.array([ z1 == z2 for z1, z2 in predYtarget ], dtype=np.float32)
precision = np.mean(precisionResults)
print("\nPrecición de la prueba de clasificación: {}%".format(str(precision*100)))

# ######## PLOTEO ##
# ##################
if plotear:
	x_objetivo = [ x for x,_y,z in all_values if z ==  1 ]
	y_objetivo = [ y for _x,y,z in all_values if z ==  1 ]
	x_trivial  = [ x for x,_y,z in all_values if z == -1 ]
	y_trivial  = [ y for _x,y,z in all_values if z == -1 ]

	min_x, min_y = np.amin(x_y_all_values,0)
	max_x, max_y = np.amax(x_y_all_values,0)

	pointAmount = 300
	termx = np.linspace(min_x, max_x, pointAmount)
	termy = np.linspace(min_y, max_y, pointAmount)

	xs, ys = np.meshgrid(termx, termy)

	Z = np.transpose(np.array([ calculoSignado(xy) for xy in zip(xs.flatten(), ys.flatten()) ], dtype=np.float32)).flatten()
	Z = Z.reshape(xs.shape)

	plt.suptitle('Linear SVM')
	plt.plot(x_objetivo, y_objetivo, 'ko', label='Objetivo')
	plt.plot(x_trivial, y_trivial, 'kx', label='Trivial')
	plt.contourf(termx, termy, Z, cmap=ListedColormap(['#ad301d', 'green']), alpha=0.8)
	plt.grid(True)
	plt.xlim([0., 1.4])
	plt.ylim([0.8, 4])
	plt.xlabel('Eje X')
	plt.ylabel('Eje Y')

	plt.show()
