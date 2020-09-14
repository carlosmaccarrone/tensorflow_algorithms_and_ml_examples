import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
################ CONSTANTES ########
####################################
batch_size = 100 
iteraciones = 200
plotear = True

################ DATOS ##
#########################
from datasetMocks import linearSoft # modulo propio

all_values = np.array([ row for row in linearSoft ], dtype=np.float32)
x_y_values = np.array([ [dat[0], dat[1]] for dat in all_values ], dtype=np.float32)
z_values = np.array([ dat[2] for dat in all_values ], dtype=np.float32)
zn_z_values = np.array([ [0,1] if dat[2] else [1,0] for dat in all_values ], dtype=np.float32)

train_data,train_labels = (x_y_values, zn_z_values)
test_data, test_labels = (x_y_values, zn_z_values)

_x_y_rows, x_y_columns = x_y_values.shape
_zn_z_rows, zn_z_columns = zn_z_values.shape

# # ######## THE MODEL TRAINING BEGINS #
# # ####################################
def modelo(puntosXY):
	calc1 = tf.matmul(puntosXY, pendiente)
	calc2 = tf.add(calc1, ordenadaAlOrigen)
	return tf.nn.softmax(calc2)

def errorDeEntropiaCruzada(imagen, imagenZNZ):
	calc5 = tf.math.log(imagen)
	calc6 = tf.multiply(imagenZNZ, calc5)
	calc7 = tf.reduce_sum(calc6)
	return tf.negative(calc7)

def algoritmo(puntosXY, imagenZNZ):
	def wrap():
		imagen = modelo(puntosXY)
		return errorDeEntropiaCruzada(imagen, imagenZNZ)
	return wrap

def model_training():
	opt = tf.keras.optimizers.SGD(learning_rate=0.01)

	for _ in range(iteraciones):
		rand_idx = np.random.choice(len(x_y_values), size=batch_size)
		puntosXY = x_y_values[rand_idx]
		imagenZNZ = zn_z_values[rand_idx]
		opt.minimize(algoritmo(puntosXY, imagenZNZ), [pendiente, ordenadaAlOrigen])

pendiente = tf.Variable(tf.zeros([x_y_columns, zn_z_columns]))
ordenadaAlOrigen = tf.Variable(tf.zeros([zn_z_columns]))
model_training()
# ######## TESTEO ##
# ##################
pend = pendiente.numpy()
ordenada = ordenadaAlOrigen.numpy()

def clasificarPuntos(puntosXY):
	calc1 = np.matmul(puntosXY, pend)
	calc2 = np.add(calc1, ordenada)
	imagen = np.divide(np.exp(calc1), np.sum(np.exp(calc2))) # softmax function
	return np.argmax(imagen, 1)

imagenResultante = clasificarPuntos(x_y_values)
predicciones = np.equal(imagenResultante, np.argmax(zn_z_values, 1))
precision = np.mean(np.array(predicciones, dtype=np.float32))
print("\nPrecisión: {}".format(str(precision*100))) 

# ######## PLOTEO ##
# ##################
if plotear:
	min_x, min_y = np.amin(x_y_values,0)
	max_x, max_y = np.amax(x_y_values,0)

	pointAmount = 300
	termx = np.linspace(min_x, max_x, pointAmount)
	termy = np.linspace(min_y, max_y, pointAmount)

	xs, ys = np.meshgrid(termx, termy)

	contents = np.c_[x_y_values, z_values]
	x_circle  = [ vec[0] for vec in contents if vec[2] == 1 ]
	y_circle  = [ vec[1] for vec in contents if vec[2] == 1 ]
	x_cross   = [ vec[0] for vec in contents if vec[2] == 0 ]
	y_cross   = [ vec[1] for vec in contents if vec[2] == 0 ]

	Z = clasificarPuntos(np.c_[xs.flatten(), ys.flatten()]);
	Z = Z.reshape(xs.shape)

	fig = plt.figure(figsize=(12,5))
	fig.suptitle('Softmax linear classifications')
	ax1 = fig.add_subplot(1, 2, 1)

	ax1.plot(x_circle, y_circle, 'ko')
	ax1.plot(x_cross, y_cross, 'kx')    
	ax1.contourf(xs, ys, Z, cmap=ListedColormap(['#ad301d', 'g']))
	ax1.grid(True)
	ax1.set_xlim([0., 1.4])
	ax1.set_ylim([0.8, 4])
	ax1.set_xlabel('Eje X')
	ax1.set_ylabel('Eje Y')
	
	ax2 = fig.add_subplot(1, 2, 2, projection='3d')
	ax2.scatter3D(x_y_values[:,0], x_y_values[:,1], c=zn_z_values[:,1], s=50, cmap=ListedColormap(['red', 'green']))

	plt.show()
