#!/usr/bin/env python
import numpy as np

def sigmoid(x):
	'''
	sigma(x) = 1 / (1 + exp(-x))
	'''
	return 1.0 / (1.0 + np.exp(-x))

def derivative_of_sigmoid(sigma_x):
	'''
	derivative of sigma(x): 
	
	dsigma(x)/dx = sigma(x) * (1 - sigma(x))

	'''
	return sigma_x * (1.0 - sigma_x)

def initialize_weights(neurons):
	''' 
	neurons is the list of neuron counts of different layers, 
	where first element is the size of inputs and the last element the size of outputs. 
	E.g [3,4,2] would mean that there are total 3 layers (len(neurons)) with 3, 4 and 5 neurons respectively
	
	This function returns the weight matrices of the hidden layers in a single list.
	The weights are Xavier initialized. 
	weights = [W_1, ..., W_n], where W_i is a weight matrix and n = len(neurons) - 1
	'''
	weights = []
	for (i,neuron_count) in enumerate(neurons):
		if i == len(neurons) - 1:
			break
		else:
			# Use Xavier initialization for weights. I.e. Var(W_i) = 1.0 / neuron_count_(i-1)
			# The np.random.normal returns an array of size (current layer neuron count))*(next layer neuron count)
			weights.append(np.random.normal(scale=1.0/neuron_count, size=(neuron_count, neurons[i+1])))
	return weights
		
		
weights = initialize_weights([2,5,1])

# Let's train the network to recognize logic XOR-gate. 
inputs = [np.array([[1,1]]),np.array([[1,0]]), np.array([[0,1]]), np.array([[0,0]])]
targets = [np.array([[0]]), np.array([[1]]), np.array([[1]]), np.array([[0]])]

learning_rate = 0.4

iterations = 15000
iteration = 0



while iteration < iterations:

	# Calculate the output of the network, given the input

	# select the input randomly
	selected_input = np.random.randint(4)
	#selected_input = 0

	# The output is the matrix multiplication of the input and the weight matrices W (+ biases, but this doesn't utilize them)
	
	derivatives = []
	# derivatives is a list of derivative matrices. Each of the derivative matrices contains
	# the derivative values of the respective layer.
	# Every derivative matrix is a diagonal matrix. This comes from the backpropagation equations.

	# we need to keep track of the outputs too for backpropagation.
	outputs = [inputs[selected_input]]


	output = inputs[selected_input]
	for W in weights:
		output = sigmoid(np.dot(output, W))		
		outputs.append(output)
		derivatives.append(np.diag(derivative_of_sigmoid(output[0,:])))
			

	print("Iteration: {}. For input {} the neural network returned {}, while target is {}".format(iteration+1, inputs[selected_input], output, targets[selected_input]))

	# for the output neuron layer: delta = (o - t) * sigmoid'(o)
	# o is the output vector, t is the target vector and sigmoid' is the derivative of sigmoid 
	

	
	current_delta = np.dot((output - targets[selected_input]), derivatives[-1])
	deltas = [None for i in range(len(derivatives))]
	deltas[-1] = current_delta

	# Backpropagation
	for layer_index in reversed(range(len(derivatives)-1)):
		current_delta = np.dot(np.dot(derivatives[layer_index], weights[layer_index+1]), current_delta)
		deltas[layer_index] = current_delta
	
	# Weight correction
	for w_index in range(len(weights)):
		weight_correction = np.transpose(-learning_rate * np.dot(deltas[w_index], outputs[w_index]))
		weights[w_index] = weights[w_index] + weight_correction

	iteration += 1

	




	




