#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:13:13 2018

@author: wentao
"""

import numpy as np
from Myhelper import *

####################################################################################
"""logistic regression in Newton method with penalty """
def logistic_regression_penalized_gradient(y, tx, max_iter):
    # set initial parameters
    gamma = 1 / (0.5 * np.max(np.linalg.eigvals(tx.T @ tx)))
    lambda_ = 0.1
    threshold = 1e-8
    losses = [100]
    # build initial weights
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)
        if iter % 100 == 0:
            print('iter:', iter,'loss:',loss)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        losses.pop(0)
    return losses[-1], w

def penalized_logistic_regression(y, tx, w, lambda_):
    # calculate loss
    loss = calculate_logistic_loss(y, tx, w) + (np.linalg.norm(w, 2) ** 2) * lambda_ / 2
    # calculate gradient
    gradient = calculate_gradient(y, tx, w) + lambda_ * w
    # calculate hessian
    hessian = calculate_hessian(y, tx, w) + lambda_
    return loss, gradient, hessian

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    # update w
    w = w - gamma * np.linalg.inv(hessian) @ gradient
    return loss, w

####################################################################################
"""logistic regression in gradient method and Newton method without penalty"""
def logistic_regression(y, tx, max_iter):
    # init parameters
    threshold = 0.0001
    w = np.zeros((tx.shape[1], 1))
    losses = [100]
    # calculate proper step size (with the reference of EE-556 slides)
    gamma = 1 / (0.5 * np.max(np.linalg.eigvals(tx.T @ tx)))
    print('stepsize:', gamma)
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if iter % 100 == 0:
            print('iter:', iter,'loss:',loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        losses.pop(0)
    return losses[-1], w

def sigmoid(t):
    return (1 / (1 + np.exp(-t)))

def calculate_logistic_loss(y, tx, w):
    L = np.sum(np.log(1 + np.exp(tx @ w)) - y * (tx @ w))
    return L

def calculate_gradient(y, tx, w):
    gradient_L = tx.T @ (sigmoid(tx @ w) - y)
    return gradient_L

def learning_by_gradient_descent(y, tx, w, gamma):
    loss = calculate_logistic_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma * gradient
    return loss, w

def calculate_hessian(y, tx, w):
    s = sigmoid(tx @ w) * (1 - sigmoid(tx @ w))
    s_diag = np.diag(s[:,0])
    hessian = tx.T @ s_diag @ tx
    return hessian

def learning_by_newton_method(y, tx, w, gamma):
    hessian = calculate_hessian(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma * np.linalg.inv(hessian) @ gradient
    loss = calculate_logistic_loss(y, tx, w)
    return loss, w

####################################################################################
"""least squares gradient descent"""
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    # set initial w
    w = initial_w
    for n_iter in range(max_iters):
        # calculate error
        e = y - tx @ w
        # calculate mse
        loss = 0.5 * np.mean((y - tx @ w) ** 2, axis = 0)
        grad = -1 * tx.T @ e / tx.shape[0]
        # update weights
        w = w - gamma * grad
        losses.append(loss)
    return losses[-1], ws[-1]

def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

####################################################################################
"""stochastic_gradient_descent"""
def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        iter = batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
        for j in iter:
                info = j
        # fetch stochastic point to calculate gradient
        y_n = info[0]
        tx_n = info[1]
        stoch_grad = compute_stoch_gradient(y_n, tx_n, w)
        # update w through the stochastic gradient update
        w = w - gamma * stoch_grad
        # calculate loss
        loss = 0.5 * np.mean((y - tx @ w) ** 2, axis = 0)
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return losses[-1], ws[-1]

def compute_stoch_gradient(y, tx, w):
    e = y -tx @ w
    stoch_grad = -1 * tx.T * e
    return stoch_grad

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    #With the reference of lab helper
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

####################################################################################
"""least squares method"""
def least_squares(y, tx):
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    return w

####################################################################################
"""ridge regression with and without cross validation"""
def ridge_regression(y, tx, lambda_):
    # calculate regularization coefficient
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)
            
def ridge_regression_cv(y, tx, lamb, k_indices, k_fold):
    w = np.zeros((tx.shape[1], k_fold))# store w for each CV set
    right_rate = []# store the correct rate of every validation
    for k in range(k_fold):
        # generate train set and test set
        y_train, x_train, y_k_test, x_k_test = cross_validation_set(y, tx, k_indices, k)
        # train model
        w[:, k] = ridge_regression(y_train, x_train, lamb)
        # predict on the test set
        test_predict_label = x_k_test @ w[:, k]
        # correcting the labels
        test_predict_label = binary_label(test_predict_label)
        # predict the labels on train set
        train_predict_label = x_train @ w[:, k]
        # correcting the labels
        train_predict_label = binary_label(train_predict_label)
        # calculate the percentage of right prediction
        # comprehensive scores: 1/3 correct rate on train set + 2/3 correct rate on test set
        comb_right_rate = (1 * calculate_right_rate(y_train, train_predict_label) + 2 * calculate_right_rate(y_k_test, test_predict_label)) / 3
        right_rate.append(comb_right_rate)
    # find the best w
    best_k = right_rate.index(max(right_rate))
    best_w = w[:, best_k]
    return best_w, max(right_rate)

