# EPFL - Machine Learning Project 1 

**`Team`:** HuoGuo_Fondue

**`Team Members`:** FENG Wentao, WANG Yunbei, ZHUANG Ying

In this project, we aim to simulate the process of discovering the Higgs particle since Higgs particles are essential for explaining why other particles have mass. Our goal is to train a binary classifier by using given training set and then use the obtained model to predict whether an event was signal (a Higgs boson) or a background (something else). We used the logistic regression model and the ridge regression model to predict on the test set and achieved the highest accuracy of 0.831 and 0.822 respectively. In the final, we chose to submit the ridge regression model since it has reasonable computing time and acceptable accuracy.

**Instrcutions**:
1. Please make sure the newest `Numpy` is installed. `NumPy` is the only required external package in this project.
2. Download `train.csv` and `test.csv` from [Kaggle competition](https://www.kaggle.com/c/epfml18-higgs/data), and put them in the same folder as `run.py`
3. Run `run.py`



The followings are some descriptions for the modules or functions we used or designed for this project.


## External modules
### `proj1_helpers.py`
In this file, the contest holders provide functions to load data, make predictions with the model as one parameter and generate the qualified submission file.

### `Myhelper.py`
There are some functions designed by us  .
- **`cross_validation_set`**: Generate cross validation sets. Input: labels, features, randomly permutated indices and k folds. Return: train set of labels, train set of features, test set of labels, test set of features.
- **`build_k_indices`**: Generate randomly permutated indecies  Input: labels, the number of folds. Return: indices
- **`calculate_right_rate`**: Calculate the percentage of corrected prediction. Input: original labels, predicted labels. Return: the right rate.
- **`binary_label`**: Generate binary labels. For example, when the predicted is greater than threshold, then we set the label 1. Input: predicted values. Return: processed labels.
- **`log_normal`**: Apply log scale and standard normalisation to the feature. When the feature is greater than 0, we use log scale. Otherwise, we use standard normalisation. Input: features. Return: processed features.
- **`calculate_mse`**: Calculate the mean square error. Input: the gap between observed value and the predicted value. Return: mean square error.
- **`calculate_mae`**: Calculate the mean average error. Input: the gap between observed value and the predicted value. Return: mean average error.
- **`compute_loss`**: Calculate the mse using observed value, features, and models. Input: labels, features, weights. Return mean square error.
- **`build_poly_feature`**: build the polynomial features for degree = 3. Input: features. Return: polynomial features.
- **`calculate_combination`**: calculate the number of results when choosing y items from x items. Input: the number of total items, the number of items people want to pick. Return: the number of all possible combinations.

## Algorithms for Regression 
### `implementations.py`
Contain the mandatory implementations of  6 regression models for this project
- **`least_squares_GD`**: Linear regression using gradient descent
- **`least_squares_SDG`**: Linear regression using stochastic gradient descent
- **`least_squares`**: Least squares regression using normal equations
- **`ridge_regression`**: Ridge regression using normal equations
- **`logistic_regression`**: using stochastic gradient descent
- **`regularized_logistic_regression`**: Regularized logistic regression

### `run.py`
Script that generates the exact CSV file submitted on Kaggle.






