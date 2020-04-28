import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
session = tf.Session()
import tensorflow.compat.v1.nn as nn

import numpy as np
import matplotlib.pyplot as plt

# Calcular una recta que se apróxime lo mejor posible a los siguientes puntos:
# (1;0), (2;2), (3;0), (4;2)

# y=ax+b
# regresión por cuadrados mínimos 0=error=Σ(aXi+b-Yi)^2
# solución a=0.4, b=0

x_vals = [1, 2, 3, 4]
y_vals = [0, 2, 0, 2]

x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_data = tf.placeholder(shape=[1], dtype=tf.float32)
a = tf.Variable(tf.random_normal(shape=[1]))
b = tf.Variable(tf.random_normal(shape=[1]))

producto = tf.multiply(a,x_data)
suma = tf.add(producto,b)

error = tf.square(suma-y_data) / (2*len(x_vals)) # error cuadrático medio

optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.025).minimize(error)

init = tf.global_variables_initializer()
session.run(init)

for epoch in range(1000):
	for (x, y) in zip(x_vals, y_vals): 
		session.run( optimizador, feed_dict={x_data: [x], y_data: [y]} )

	# print( "a="+str(session.run(a))+", b="+str(session.run(b)) )

final_a = session.run(a)
final_b = session.run(b)

# Los valores finales no representan la mejor recta posible pero sí es una sucesión convergente.
print("\nEcuación de la recta: y="+str(np.round(final_a,1)[0])+"x+"+str(np.round(final_b,1)[0])+"\n")

# Evaluación del modelo
x_vals_test = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 16, 17, 18, 19, 20, 21, 22, 23 ,24]
y_vals_test = [1.6, 3.6, 1.6, 3.6, 3.2, 5.2, 3.2, 5.2, 4.8, 6.8, 4.8, 6.8, 6.4, 8.4, 6.4, 8.4, 8, 10, 8, 10]

eje_x = tf.placeholder(shape=[1, None], dtype=tf.float32)
eje_y = tf.placeholder(shape=[1, None], dtype=tf.float32)
n = tf.placeholder(dtype=tf.float32)

# Se trata de apróximar, por ello no se puede valúar la función pero, se puede 
# calcular la precisión basandose en que el error debe ser siempre próximo a 0.
calc1 = tf.multiply(eje_x, final_a)
calc2 = tf.add(calc1, final_b)
calc3 = tf.square(calc2-eje_y) / (2*n)
calc4 = tf.squeeze( tf.round( calc3 ) )
calc5 = tf.cast(tf.equal(0.0, calc4), tf.float32)
precision = tf.reduce_mean(calc5)

train_prec = session.run(precision, feed_dict={eje_x: [x_vals], eje_y: [y_vals], n: len(x_vals)})
test_prec = session.run(precision, feed_dict={eje_x: [x_vals_test], eje_y: [y_vals_test], n: len(x_vals_test)})

print("Precisión en entrenamiento: "+str(train_prec*100)+"%")
print("Precisión en testeo: "+str(test_prec*100)+"%")

# Ploteo del training
imagen_recta = []
for i in range(-1,30):
    imagen_recta.append(final_a*i+final_b)
x_vals.extend(x_vals_test)
y_vals.extend(y_vals_test)
plt.plot(x_vals, y_vals, 'ro', label='puntos', alpha=0.8)
for (x, y) in zip(x_vals, y_vals):
    plt.text(x-0.2, y-0.2, '({};{})'.format(x, y))
plt.plot(range(-1,30), imagen_recta, label='recta') 
plt.suptitle('Recta que mejor ajusta', fontsize=18) 
plt.xlabel("Eje x")
plt.ylabel("Eje y")
plt.legend(borderpad=1) 
plt.xlim([-1, 26])
plt.ylim([-1, 12])
plt.grid()
plt.show()

