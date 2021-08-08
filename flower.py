# Planar data classification with one hidden layer
import numpy as np
import copy
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# define the neural network structure
def layer_sizes(X, Y):
	"""
	Arguments:
	X -- input dataset of shape (input size, number of examples)
	Y -- labels of shape (output size, number of examples)

	Returns:
	n_x -- the size of the input layer
	n_h -- the size of the hidden layer
	n_y -- the size of the output layer
	"""
	n_x = X.shape[0]
	n_h = 4
	n_y = Y.shape[0]
	return (n_x, n_h, n_y)

#initialize the model's parameters
def initialize_parameters(n_x, n_h, n_y):
	"""
	Arguments:
	n_x -- size of the input layer
	n_h -- size of the hidden layer
	n_y -- size of the output layer

	Returns:
	params -- python dictionary containing your parameters:
			W1 -- weight matrix of shape (n_h, n_x)
			b1 -- bias vector of shape (n_h, 1)
			W2 --weight matrix of shape (n_y, n_h)
			b2 -- bias vector of shape (n_y, 1)
	"""
	W1 = np.random.randn(n_h, n_x) * 0.01
	b1 = np.zeros((n_h, 1))
	W2 = np.random.randn(n_y, n_h) * 0.01
	b2 = np.zeros((n_y, 1))

	parameters = {
	"W1": W1,
	"b1": b1,
	"W2": W2,
	"b2": b2
	}
	return parameters

# forward propagation function
def forward_propagation(X, parameters):
	"""
	Arguments:
	X -- input data of size (n_x, m)
	parameters -- python dictionary containing your parameters (output of initialization function)

	Returns:
	A2 -- the sigmoid output of the second activation
	cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
	"""
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	#forward propagation
	Z1 = np.dot(W1, X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = sigmoid(Z2)

	assert(A2.shape == (1, X.shape[1]))

	cache = {
	"Z1": Z1,
	"A1": A1,
	"Z2": Z2,
	"A2": A2
	}

	return A2, cache

# compute the cost 
def compute_cost(A2, Y):
	"""
	Computes the cross-entropy cost

	Arguments:
	A2 -- the sigmoid output of the second activation of shape (1, number of examples)
	Y -- "true" labels vector of shape (1, number of examples)

	Returns:
	cost -- cross-entropy cost
	"""
	# number of examples
	m = X.shape[1]
	logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
	cost = (-1 / m) * np.sum(logprobs)
	cost = float(np.squeeze(cost))

	return cost

# backward propagation
def backward_propagation(parameters, cache, X, Y):
	"""
	Implement the backward propagation

	Arguments:
	parameters -- python dictionary containing our parameters
	cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
	X -- input data of shape (2, number of examples)
	Y -- 'True' labels vector of shape (1, number of exmaples)

	Returns:
	grads -- python dictionary containing your gradients with respect to different parameters
	"""
	# number of examples
	m = X.shape[1]
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	A1 = cache["A1"]
	A2 = cache["A2"]
	# backward propagation
	dZ2 = A2 - Y
	dW2 = (1 / m) * np.dot(dZ2, A1.T)
	db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
	dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
	dW1 = (1 / m) * np.dot(dZ1, X.T)
	db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

	grads = {
	"dW1": dW1,
	"db1": db1,
	"dW2": dW2,
	"db2": db2
	}

	return grads

# update parameters
def update_parameters(parameters, grads, learning_rate=1.2):
	"""
	Updates parameters using the gradient descent update rule

	Arguments:
	parameters -- python dictionary containing your parameters
	grads -- python dictionary containing your gradients

	Returns:
	parameters -- python dictionary containing your updated parameters
	"""
	# copy parameters
	W1 = copy.deepcopy(parameters["W1"])
	b1 = copy.deepcopy(parameters["b1"])
	W2 = copy.deepcopy(parameters["W2"])
	b2 = copy.deepcopy(parameters["b2"])
	# retrieve gradients
	dW1 = grads["dW1"]
	db1 = grads["db1"]
	dW2 = grads["dW2"]
	db2 = grads["db2"]
	# update rule
	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	parameters = {
	"W1": W1,
	"b1": b1,
	"W2": W2,
	"b2": b2
	}

	return parameters

# neural network model
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
	"""
	Arguments:
	X -- dataset of shape (2, number of examples)
	Y -- labels of shape (1, number of examples)
	n_h -- size of the hidden layer
	num_iterations -- number of iterations in gradient descent loop
	print_cost -- if True, print the cost every 1000 iterations

	Returns:
	parameters -- parameters learnt by the model. They can then be used to predict.
	"""
	n_x = layer_sizes(X, Y)[0]
	n_y = layer_sizes(X, Y)[2]
	# initialize parameters
	parameters = initialize_parameters(n_x, n_h, n_y)
	# loop gradient descent
	for i in range(0, num_iterations):
		# forward propagation
		A2, cache = forward_propagation(X, parameters)
		# cost
		cost = compute_cost(A2, Y)
		# backward propagation
		grads = backward_propagation(parameters, cache, X, Y)
		# update parameters
		parameters = update_parameters(parameters, grads)

		if (print_cost and i % 1000 == 0):
			print('Cost after iteration %i: %f' %(i, cost))

	return parameters

# predictions
def predict(parameters, X):
	"""
	Using the learned parameters, predicts a class for each example in X

	Arguments:
	parameters -- python dictionary containing your parameters
	X -- input data of size (n_x, m)

	Returns:
	predictions -- vector of predictions of our model (red: 0 / blue: 1)
	"""
	# computes probabilities using forward propagation and classifies to 0/1 using 0.5 as the threshold
	A2, cache = forward_propagation(X, parameters)
	predictions = A2 > 0.5

	return predictions

# test the model on the planar dataset
X, Y = load_planar_dataset()
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
# plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

# print accuracy
predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + "%")

# tuning hidden layer size
plt.figure(figsize = (16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
	plt.subplot(5, 2, i + 1)
	plt.title('Hidden layer of size %d' % n_h)
	parameters = nn_model(X, Y, n_h, num_iterations=5000)
	plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
	predictions = predict(parameters, X)
	accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
	print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

plt.show()