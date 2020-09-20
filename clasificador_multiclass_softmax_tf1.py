import os
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.disable_v2_behavior()
session = tf.Session()

## Éste código es beta, puede y será mejorado en un futuro.

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

################ CONSTANTES ########
####################################
batch_size = 100 
numNodos, iteraciones = (5, 2000)  ## (3, 1000) # según la configuración el modelo servirá más o menos
plotear = True

# ################ DATOS ##
# #########################
from datasetMocks import multiLabelArr
allTodo = np.array([ row for row in multiLabelArr ], dtype=np.float32)
x_y_all_values = np.array([ [x,y] for x,y,_z in allTodo ], dtype=np.float32)
z_all_values = np.array([ z for _x,_y,z in allTodo ], dtype=np.float32)
zn_z_values = np.array([ [1,0,0,0] if z == 0 else [0,1,0,0] if z == 1 else [0,0,1,0] if z == 2 else [0,0,0,1] if z == 3 else z for z in z_all_values ], dtype=np.float32)

_x_y_rows, x_y_columns = x_y_all_values.shape
_zn_z_rows, zn_z_columns = zn_z_values.shape

# # ######## THE MODEL TRAINING BEGINS #
# # ####################################
puntosXY = tf.placeholder(shape=[None, 2], dtype=tf.float32)
imagenZNZ = tf.placeholder(shape=[None, zn_z_columns], dtype=tf.float32)
pendiente_1 = tf.Variable(tf.random_normal(shape=[x_y_columns, numNodos]))
pendiente_2 = tf.Variable(tf.random_normal(shape=[numNodos, zn_z_columns]))
ordenada_1 = tf.Variable(tf.zeros([1, numNodos]))
ordenada_2 = tf.Variable(tf.zeros([1, zn_z_columns]))

calc1 = tf.matmul(puntosXY, pendiente_1)
calc2 = tf.add(calc1, ordenada_1)
primeraCapa = tf.nn.tanh(calc2)

calc3 = tf.matmul(primeraCapa, pendiente_2)
calc4 = tf.add(calc3, ordenada_2)
imagen = tf.nn.softmax(calc4)

# loss function
calc5 = tf.log(imagen)
calc6 = tf.multiply(imagenZNZ, calc5)
calc7 = tf.reduce_sum(calc6)
cross_entropy = tf.negative(calc7)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
session.run(init)	

for step in range(iteraciones):
	rand_idx = np.random.choice(len(x_y_all_values), size=batch_size)
	batch_data = x_y_all_values[rand_idx]
	batch_labels = zn_z_values[rand_idx]
	session.run(train_step, feed_dict={puntosXY: batch_data, imagenZNZ: batch_labels})

# # ######## TESTEO ##
# # ##################
pend1 = session.run(pendiente_1, feed_dict={puntosXY: x_y_all_values})
pend2 = session.run(pendiente_2, feed_dict={puntosXY: x_y_all_values})
ord1 = session.run(ordenada_1, feed_dict={puntosXY: x_y_all_values})
ord2 = session.run(ordenada_2, feed_dict={puntosXY: x_y_all_values})

def clasificarPuntos(puntosXY):
	calc1 = np.matmul(puntosXY, pend1)
	calc2 = np.add(calc1, ord1)
	primeraCapa = np.tanh(calc2)

	calc3 = np.matmul(primeraCapa, pend2)
	calc4 = np.add(calc3, ord2)
	imagen = np.divide(np.exp(calc4), np.sum(np.exp(calc4))) # softmax function
	return np.argmax(imagen, 1)

imagenResultante = clasificarPuntos(x_y_all_values)
predicciones = np.equal(imagenResultante, np.argmax(zn_z_values, 1))
precision = np.mean(np.array(predicciones, dtype=np.float32))
print("\nPrecisión: {}".format(str(precision*100))) 

# # ######## PLOTEO ##
# # ##################
if plotear:
	min_x, min_y = np.amin(x_y_all_values,0)-1
	max_x, max_y = np.amax(x_y_all_values,0)+1

	pointAmount = 300
	termx = np.linspace(min_x, max_x, pointAmount)
	termy = np.linspace(min_y, max_y, pointAmount)

	xs, ys = np.meshgrid(termx, termy)

	contents = np.c_[x_y_all_values, z_all_values]
	x_first_class  = [ x for x, _y, z in contents if z == 0 ]
	y_first_class  = [ y for _x, y, z in contents if z == 0 ]
	x_second_class = [ x for x, _y, z in contents if z == 1 ]
	y_second_class = [ y for _x, y, z in contents if z == 1 ]
	x_third_class  = [ x for x, _y, z in contents if z == 2 ]
	y_third_class  = [ y for _x, y, z in contents if z == 2 ]
	x_four_class   = [ x for x, _y, z in contents if z == 3 ]
	y_four_class   = [ y for _x, y, z in contents if z == 3 ]

	Z = clasificarPuntos(np.c_[xs.flatten(), ys.flatten()]);
	Z = Z.reshape(xs.shape)

	plt.suptitle('Softmax non-linear multiclass classifications')

	plt.plot(x_first_class, y_first_class, 'ko')
	plt.plot(x_second_class, y_second_class, 'kx')
	plt.plot(x_third_class, y_third_class, 'kv')
	plt.plot(x_four_class, y_four_class, 'k^')
	plt.contourf(xs, ys, Z, cmap=ListedColormap(['#ad301d', 'b', 'g', 'c']), alpha=0.8)
	plt.grid(True)
	plt.xlabel('Eje X')
	plt.ylabel('Eje Y')

	plt.show()

## Éste código es beta, puede y será mejorado en un futuro.
