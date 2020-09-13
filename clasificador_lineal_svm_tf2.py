import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

################ DATOS ##
#########################
from datasetMocks import linearSVM # modulo propio

all_values = np.array([ row for row in linearSVM ], dtype=np.float32)
x_y_all_values = np.array([ [dat[1], dat[2]] for dat in all_values ], dtype=np.float32)
z_all_values = np.array([ dat[0] for dat in all_values ], dtype=np.float32)

x_y_vals_train = np.array([ row for row in x_y_all_values ], dtype=np.float32)
z_vals_train = np.array([ row for row in z_all_values ], dtype=np.float32)

dataLen = len(x_y_vals_train)
test_idx = np.random.choice((dataLen-14), round(dataLen*0.2), replace=False)
x_y_vals_test = x_y_all_values[test_idx]
z_vals_test = z_all_values[test_idx]

# ######## THE MODEL TRAINING BEGINS #
# ####################################
def perdidaSVMLineal(puntos, clasificaciones):
	def algoritmo():
		output = tf.add(tf.matmul(puntos, pendiente), ordenadaAlOrigen)
		norma_l2 = tf.reduce_sum(tf.square(pendiente))
		alpha = tf.constant([0.1]) # like learning_rate or batch_size, can be changed
		loss = tf.reduce_sum(tf.maximum(0., tf.subtract(1., tf.multiply(output, clasificaciones))))
		return tf.add(loss, tf.multiply(alpha, norma_l2))
	return algoritmo

def model_training():
	opt = tf.keras.optimizers.SGD(learning_rate=0.01)

	for _ in range(2000):
		rand_idx = np.random.choice(dataLen, size=70)
		puntos = x_y_vals_train[rand_idx]
		clasificaciones = np.transpose([z_vals_train[rand_idx]])
		opt.minimize(perdidaSVMLineal(puntos, clasificaciones), [pendiente, ordenadaAlOrigen])

pendiente = tf.Variable(tf.random.normal(shape=[2,1]))
ordenadaAlOrigen = tf.Variable(tf.random.normal(shape=[1,1]))
model_training()

# # ######## TESTEO ##
# # ##################
pend = pendiente.numpy()
ordOrigen = ordenadaAlOrigen.numpy()
calculoSignado = lambda x, y: np.sign(np.add(np.matmul([x,y], pend), ordOrigen)).flatten()
predicciones = [ calculoSignado(x, y) for x, y in x_y_vals_test ]
precision = np.mean(np.array(np.equal(predicciones, z_vals_test), dtype=np.float32))
print("\nPrecición de la prueba de clasificación: {}%".format(str(precision*100)))

# ######## PLOTEO ##
# ##################
x_objetivo = [ elem[1] for elem in all_values if elem[0] ==  1 ]
y_objetivo = [ elem[2] for elem in all_values if elem[0] ==  1 ]
x_trivial  = [ elem[1] for elem in all_values if elem[0] == -1 ]
y_trivial  = [ elem[2] for elem in all_values if elem[0] == -1 ]

min_x, min_y = np.amin(x_y_all_values,0)
max_x, max_y = np.amax(x_y_all_values,0)

pointAmount = 300
termx = np.linspace(min_x, max_x, pointAmount)
termy = np.linspace(min_y, max_y, pointAmount)

xs, ys = np.meshgrid(termx, termy)

Z = np.array([calculoSignado(x, y) for x, y in zip(xs.flatten(), ys.flatten())])
Z = Z.reshape(xs.shape)

fig = plt.figure(figsize=(12,5))
fig.suptitle('Support vector machine')

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x_objetivo, y_objetivo, 'ko', label='Objetivo')
ax1.plot(x_trivial, y_trivial, 'kx', label='Trivial')
ax1.contourf(termx, termy, Z, cmap=ListedColormap(['red', 'green']))
ax1.grid(True)
ax1.set_xlim([0., 1.4])
ax1.set_ylim([0.8, 4])
ax1.set_xlabel('Eje X')
ax1.set_ylabel('Eje Y')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter3D(xs, ys, Z, c=Z, cmap=ListedColormap(['red', 'green']))
plt.show()
