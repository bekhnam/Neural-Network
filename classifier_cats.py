import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# build sigmoid function
def sigmoid(z):
	"""
	Compute the sigmoid of z
	
	Arguments:
	z -- A scalar or numpy array of any size

	Return:
	s -- sigmoid(z)
	"""
	return 1 / (1 + np.exp(-z))

# initialized function for w and b as a vector of zeros
def initialize_with_zeros(dim):
	"""
	This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

	Argument:
	dim -- size of the w vector we want (or number of parameters in this case)

	Returns:
	w -- initialized vector of shape (dim, 1)
	b -- initialized scalar (corresponds to the bias) of type float
	"""
	w = np.zeros((dim, 1))
	b = 0.
	return w, b

# backward and forward propagation function
def propagate(w, b, X, Y):
	"""
	Implement the cost function and its gradient for the propagation

	Arguments:
	w -- weights, a numpy array of size (num_px * num_px * 3, 1)
	b -- bias, a scalar
	X -- data of size (num_px * num_px * 3, number of examples)
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

	Returns:
	cost -- nagative log-likelihood cost for logistic regression
	dw -- gradient of the loss with respect to w, thus same shape as w
	db -- gradient of the loss with respect to b, thus same shape as b
	"""
	# number of examples
	m = X.shape[1]
	# forward propagation
	# sigmoid function
	A = sigmoid(np.dot(w.T, X) + b)
	# cost function
	cost = (-1 / m) * (np.sum((Y * np.log(A)) + (1 - Y) * np.log(1 - A)))
	# backward propagation
	dw = (1 / m) * np.dot(X, (A - Y).T)
	db = (1 / m) * np.sum(A - Y)

	cost = np.squeeze(np.array(cost))

	grads = {"dw": dw, "db": db}

	return grads, cost

# optimization function
def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
	"""
	This function optimizes w and b by running a gradient descent algorithm

	Arguments:
	w -- weights, a numpy array of size (num_px * num_px * 3, 1)
	b -- bias , a scalar
	X -- data of shape (num_px * num_px * 3, number of examples)
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
	num_iterations -- number of iterations of the optimization loop
	learning_rate -- learning rate of the gradient descent update rule
	print_cost -- True to print the loss every 100 steps

	Returns:
	params -- dictionary containing the weights w and bias b
	grads -- dictionary containing the gradient of the weights and bias with respect to the cost function
	costs -- list of all costs computed during the optimization, this will be used to plot the learning curve.
	"""

	w = copy.deepcopy(w)
	b = copy.deepcopy(b)

	costs = []

	for i in range(num_iterations):
		grads, cost = propagate(w, b, X, Y)
		dw = grads["dw"]
		db = grads["db"]
		# update rule
		w = w - learning_rate * dw
		b = b - learning_rate * db
		#record the costs
		if i % 100 == 0:
			costs.append(cost)
			if print_cost:
				print("Cost after iteration %i: %f" %(i, cost))

	params = {"w": w, "b": b}
	grads = {"dw": dw, "db": db}

	return params, grads, costs 

# prediction function
def predict(w, b, X):
	"""
	Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

	Arguments:
	w -- weights, a numpy array of size (num_px * num_px * 3, 1)
	b -- bias, a scalar
	X -- data of size (num_px * num_px * 3, number of examples)

	Returns:
	Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
	"""
	# number of examples
	m = X.shape[1]
	Y_prediction = np.zeros((1, m))
	w = w.reshape(X.shape[0], 1)
	# sigmoid function
	A = sigmoid(np.dot(w.T, X) + b)
	for i in range(A.shape[1]):
		# convert probabilities A[0, i] to actual predictions p[0, i]
		if A[0, i] > 0.5:
			Y_prediction[0, i] = 1
		else:
			Y_prediction[0, i] = 0

	return Y_prediction

# model function 
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
	"""
	Builds the logistic regression model by calling the function you've implemented previously

	Arguments:
	X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
	Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
	X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
	Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
	num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
	learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
	print_cost -- set to True to print the cost every 100 iterations

	Return:
	d -- dictionary containing information about the model.
	"""

	# initialize w and b with zeros
	w, b = initialize_with_zeros(X_train.shape[0])
	# params, grads, costs
	params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
	# retrieve parameters w and b from dictionary "params"
	w = params["w"]
	b = params["b"]
	# predict test/train set examples
	Y_prediction_train = predict(w, b, X_train)
	Y_prediction_test = predict(w, b, X_test)
	# print train/test Errors
	if print_cost:
		print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
		print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

	d = {"costs": costs,
		"Y_prediction_test": Y_prediction_test,
		"Y_prediction_train": Y_prediction_train,
		"w": w,
		"b": b,
		"learning_rate": learning_rate,
		"num_iterations": num_iterations} 

	return d

# loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# standardize dataset
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# build logistic regression model
logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# example of a picture
index = 6
num_px = train_set_x_orig.shape[1]
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print('\n' + "Y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[int(logistic_regression_model['Y_prediction_test'][0, index])].decode("utf-8") + "\" picture.")

# plot learning curve (with costs)
# costs = np.squeeze(logistic_regression_model['costs'])
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
# plt.show()

# choice of learning rate
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for lr in learning_rates:
	print("Training a model with learning rate: " + str(lr))
	models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1200, learning_rate=lr, print_cost=False)
	print('\n' + "--------------------------------------------------" + '\n')

for lr in learning_rates:
	plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc = 'upper center', shadow = True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()