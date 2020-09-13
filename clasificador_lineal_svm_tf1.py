import os
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.disable_v2_behavior()
session = tf.Session()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

################ DATOS ##
#########################
from datasetMocks import linearSVM # modulo propio

all_values = np.array([ row for row in linearSVM ])
x_all_values = np.array([ [dat[1], dat[2]] for dat in all_values ])
y_all_values = np.array([ dat[0] for dat in all_values ])

x_vals_train = np.array([ row for row in x_all_values ])
y_vals_train = np.array([ row for row in y_all_values ])

dataLen = len(x_vals_train)
test_idx = np.random.choice((dataLen-14), round(dataLen*0.2), replace=False)
x_vals_test = x_all_values[test_idx]
y_vals_test = y_all_values[test_idx]

# ######## THE MODEL TRAINING BEGINS #
# ####################################

batch_size = 70

puntos = tf.placeholder(shape=[None, 2], dtype=tf.float32)
clasificacion = tf.placeholder(shape=[None, 1], dtype=tf.float32)
pendiente = tf.Variable(tf.random_normal(shape=[2,1]))
ordenadaAlOrigen = tf.Variable(tf.random_normal(shape=[1,1]))

output = tf.add(tf.matmul(puntos, pendiente), ordenadaAlOrigen)

norma_l2 = tf.reduce_sum(tf.square(pendiente))
alpha = tf.constant([0.1])
loss = tf.reduce_sum(tf.maximum(0., tf.subtract(1., tf.multiply(output, clasificacion))))
svm_loss = tf.add(loss, tf.multiply(alpha, norma_l2))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)

init = tf.global_variables_initializer()
session.run(init)

for _ in range(2000):
	rand_idx = np.random.choice(dataLen, size=batch_size)
	puntos_aleatorios = x_vals_train[rand_idx]
	clasificaciones_aleatorias = np.transpose([y_vals_train[rand_idx]])
	session.run(train_step, feed_dict={puntos: puntos_aleatorios, clasificacion: clasificaciones_aleatorias})

# ######## TESTEO ##
# ##################
pend = session.run(pendiente)
ordOrigen = session.run(ordenadaAlOrigen)
calculoSignado = lambda x, y: np.sign(np.add(np.matmul([x,y], pend), ordOrigen)).flatten()
predicciones = [ calculoSignado(x, y)[0] for x, y in x_vals_test ]
precision = np.mean(np.array(np.equal(predicciones, y_vals_test), dtype=np.float32))
print("\nPrecición de la prueba de clasificación: {}%".format(str(precision*100)))

# ######## PLOTEO ##
# ##################
x_objetivo = [ elem[1] for elem in all_values if elem[0] ==  1 ]
y_objetivo = [ elem[2] for elem in all_values if elem[0] ==  1 ]
x_trivial  = [ elem[1] for elem in all_values if elem[0] == -1 ]
y_trivial  = [ elem[2] for elem in all_values if elem[0] == -1 ]

min_x, min_y = np.amin(x_all_values,0)
max_x, max_y = np.amax(x_all_values,0)

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
