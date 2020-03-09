"""
Author: Yeshu Li
The Python program has been tested under macOS Mojava Version 10.14.3 and Ubuntu 18.04.

The file paths are hard-coded in the code for my convenience. There are 4 features in crf.py file.

1. p2a function computes the required log-likelihood and stores the required gradients in gradients.txt.
2. p2b function computes the optimal parameter by using L-BFGS-B optimization method, outputs the final objective function value and stores the optimal parameter in solution.txt.
3. checkGrad function checks the gradients against finite differences.


"""

# import time
import math
import numpy as np
import torch


def computeAllDotProduct(w, word):
   
	data, label = word
	dots = np.dot(w, data.transpose())

	return dots

def logTrick(numbers):

	if len(numbers.shape) == 1:
		M = np.max(numbers)
		return M + np.log(np.sum(np.exp(numbers - M)))
	else:
		M = np.max(numbers, 1)
		return M + np.log(np.sum(np.exp((numbers.transpose() - M).transpose()), 1))

def logPYX(word, w, T, alpha, dots):

	data, label = word
	m = len(label)
	res = sum([dots[label[i], i] for i in range(m)]) + sum([T[label[i], label[i + 1]] for i in range(m - 1)])
	logZ = logTrick(dots[:, m - 1] + alpha[m - 1, :])
	res -= logZ

	return res

def computeDP(word, w, T, dots):

	data, label = word
	m = len(label)
	alpha = np.zeros((m, K))
	for i in range(1, m):
		alpha[i] = logTrick(np.tile(dots[:, i - 1] + alpha[i - 1, :], (K, 1)) + T.transpose())
	beta = np.zeros((m, K))
	for i in range(m - 2, -1, -1):
		beta[i] = logTrick(np.tile(dots[:, i + 1] + beta[i + 1, :], (K, 1)) + T)

	return alpha, beta

def computeMarginal(word, w, T, alpha, beta, dots):

	data, label = word
	m = len(label)
	p1 = np.zeros((m, K))
	for i in range(m):
		p1[i] = alpha[i, :] + beta[i, :] + dots[:, i]
		p1[i] = np.exp(p1[i] - logTrick(p1[i]))
	p2 = np.zeros((m - 1, K, K))
	for i in range(m - 1):
		p2[i] = np.tile(alpha[i, :] + dots[:, i], (K, 1)).transpose() + np.tile(beta[i + 1, :] + dots[:, i + 1], (K, 1)) + T
		p2[i] = np.exp(p2[i] - logTrick(p2[i].flatten()))

	return p1, p2

def computeGradientWy(word, p1):

	data, label = word
	m = len(label)
	cof = np.zeros((K, m))
	for i in range(m):
		cof[label[i], i] = 1
	cof -= p1.transpose()
	res = np.dot(cof, data)

	return res

def computeGradientTij(word, p2):

	data, label = word
	m = len(label)
	res = np.zeros(p2.shape)
	for i in range(m - 1):
		res[i, label[i], label[i + 1]] = 1
	res -= p2
	res = np.sum(res, 0)
   
	return res

def crfFuncGrad(params, dataset, C, num_labels, embed_dim):

	w = np.array(params[ : embed_dim * num_labels]).reshape(num_labels, embed_dim)
	T = np.array(params[embed_dim * num_labels : ]).reshape(num_labels, num_labels)

	meandw = np.zeros((num_labels, embed_dim))
	meandT = np.zeros((num_labels, num_labels))

	for word in dataset:

		dots = computeAllDotProduct(w, word)
		alpha, beta = computeDP(word, w, T, dots)
		p1, p2 = computeMarginal(word, w, T, alpha, beta, dots)

		dw = computeGradientWy(word, p1)
		dT = computeGradientTij(word, p2)

		meandw += dw
		meandT += dT

	meandw /= len(dataset)
	meandT /= len(dataset)

	meandw *= (-C)
	meandT *= (-C)

	meandw += w
	meandT += T

	gradients = np.concatenate((meandw.flatten(), meandT.flatten()))

	return gradients

def obj_func(params, dataset, C, num_labels, embed_dim) :
	w = np.array(params[ : embed_dim * num_labels]).reshape(num_labels, embed_dim)
	T = np.array(params[embed_dim * num_labels : ]).reshape(num_labels, num_labels)
	meanLogPYX = 0
	for word in dataset :
		dots = computeAllDotProduct(w, word)
		alpha, beta = computeDP(word, w, T, dots)
		meanLogPYX += logPYX(word, w, T, alpha, dots)
	meanLogPYX /= len(dataset)

	objValue = -C * meanLogPYX + 0.5 * np.sum(w ** 2) + 0.5 * np.sum(T ** 2)
	return objValue

