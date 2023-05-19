from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

batch_size = 200
gamma = -20.
plotear = True
########## DATOS ###################
####################################
(x_y_all_values, z_all_values)  = datasets.make_moons(n_samples=50, noise=.1) ## switch a medialunas
# (x_y_all_values, z_all_values) = datasets.make_circles(n_samples=100, factor=0.5, noise=0.1) ## switch a circunferencias

z_all_values = np.array([1 if z else -1 for z in z_all_values], dtype=np.float32)

rand_idx = np.random.choice(len(x_y_all_values), size=batch_size)
x_y_test = x_y_all_values[rand_idx]
z_test = np.transpose([ z_all_values[rand_idx] ])


# ######## TESTEO ##
# ##################
def calculoPredicciones(x_y_input, z_input, pointsToBeClassified, gammaConst):
	sumUnitToSquares = lambda data: np.sum(np.square(data), 1)

	reshapedX_Y_input = sumUnitToSquares(x_y_input).reshape([-1, 1]) # returns one column array
	resultantedPoints = sumUnitToSquares(pointsToBeClassified)
	transposedPoints = np.transpose(pointsToBeClassified)
	transposedResultantedPoints = np.transpose(resultantedPoints)
	transposedZ_Input = np.transpose(z_input)
	calc1 = np.matmul(x_y_input, transposedPoints)
	calc2 = np.multiply(2., calc1)
	calc3 = np.subtract(reshapedX_Y_input, calc2)
	calc4 = np.add(calc3, transposedResultantedPoints)
	calc5 = np.multiply(gammaConst, calc4)
	kernel = np.exp(calc5)
	output = np.matmul(transposedZ_Input, kernel)
	return np.sign(output)


def refactor(pointsToBeClassified, z_input):
	sumUnitToSquares = lambda data: np.sum(np.square(data), 1)
	givenPoints = sumUnitToSquares(pointsToBeClassified).reshape([-1, 1])
	transposedXYData = np.transpose(pointsToBeClassified)
	transposedGivenPoints = np.transpose(givenPoints)

	calc1 = np.matmul(pointsToBeClassified, transposedXYData)
	calc2 = np.multiply(2., calc1)
	calc3 = np.subtract(givenPoints, calc2)
	calc4 = np.add(calc3, transposedGivenPoints)
	calc5 = np.multiply(gamma, calc4)
	kernel = np.exp(calc5)
	output = np.matmul(z_input, kernel)
	return np.sign(output)

predicciones = calculoPredicciones(x_y_test, z_test, x_y_test, gamma)
predYtarget = np.c_[predicciones.flatten(), z_test]
precisionResults = np.array([ row[0] == row[1] for row in predYtarget ], dtype=np.float32)
precision = np.mean(precisionResults)
print("\nPrecición de la prueba de clasificación: {}%".format(str(precision*100)))

# ######## PLOTEO ##
# ##################
if plotear:
	contents = np.c_[x_y_all_values, z_all_values]
	x_circle  = [ vec[0] for vec in contents if vec[2] ==  1 ]
	y_circle  = [ vec[1] for vec in contents if vec[2] ==  1 ]
	x_cross   = [ vec[0] for vec in contents if vec[2] == -1 ]
	y_cross   = [ vec[1] for vec in contents if vec[2] == -1 ]

	min_x, min_y = np.amin(x_y_all_values,0)-1
	max_x, max_y = np.amax(x_y_all_values,0)+1

	pointAmount = 30
	termx = np.linspace(min_x, max_x, pointAmount)
	termy = np.linspace(min_y, max_y, pointAmount)

	xs, ys = np.meshgrid(termx, termy)

	puntosAClasificar = np.c_[xs.flatten(), ys.flatten()]
	# Z_clasificaciones = refactor(puntosAClasificar, z_all_values)
	Z_clasificaciones = calculoPredicciones(x_y_all_values, z_all_values, puntosAClasificar, gamma)
	Z_clasificaciones = Z_clasificaciones.reshape(xs.shape)

	fig = plt.figure(figsize=(12,5))
	fig.suptitle('Support vector machine')

	ax1 = fig.add_subplot(1, 2, 1)
	ax1.plot(x_circle, y_circle, 'ko')
	ax1.plot(x_cross, y_cross, 'kx')
	ax1.contourf(termx, termy, Z_clasificaciones, cmap=ListedColormap(['#ad301d', 'g']))
	ax1.grid(True)
	ax1.set_xlabel('Eje X')
	ax1.set_ylabel('Eje Y')

	ax2 = fig.add_subplot(1, 2, 2, projection='3d')
	ax2.scatter3D(x_y_all_values[:,0], x_y_all_values[:,1], c=z_all_values, s=50, cmap=ListedColormap(['red', 'green']))

	plt.show()
