# EPFL - Machine Learning Project 1 

**`Team`:** HuoGuo_Fondue

**`Team Members`:** FENG Wentao, WANG Yunbei, ZHUANG Ying

In this project, we aim to simulate the process of discovering the Higgs particle since Higgs particles are essential for explaining why other particles have mass. Our goal is to train a binary classifier by using given training set and then use the obtained model to predict whether an event was signal (a Higgs boson) or a background (something else). We used the logistic regression model and the ridge regression model to predict on the test set and achieved the highest accuracy of 0.831 and 0.822 respectively. In the final, we chose to submit the ridge regression model since it has reasonable computing time and acceptable accuracy.

**Instructions**:
1. Please make sure that the newest `Numpy` is installed. `NumPy` is the only required external package in this project.
2. Download `train.csv` and `test.csv` from [Kaggle competition](https://www.kaggle.com/c/epfml18-higgs/data), and put them in the same folder as `run.py`
3. Run `run.py`



The followings are some descriptions for the modules or functions we used or designed for this project.


## External modules
### `proj1_helpers.py`
In this file, the contest holders provide functions to load data, make predictions with the model as one input parameter and generate the qualified submission file.

### `Myhelper.py`
There are some functions designed by us.
- **`cross_validation_set`**: Generate cross-validation sets. `Input`: labels, features, randomly permutated indices, and k folds. `Return`: train set of labels, train set of features, test set of labels, test set of features.
- **`build_k_indices`**: Generate randomly permutated indices  `Input`: labels, the number of folds. `Return`: indices
- **`calculate_right_rate`**: Calculate the percentage of corrected prediction. Input: original labels, predicted labels. `Return`: the right rate.
- **`binary_label`**: Generate binary labels. For example, when the predicted is greater than the threshold, then we set the label 1. `Input`: predicted values. `Return`: processed labels.
- **`log_normal`**: Apply log scale and standard normalisation to the feature. When the feature is greater than 0, we use log scale. Otherwise, we use standard normalisation. `Input`: features. `Return`: processed features.
- **`calculate_mse`**: Calculate the mean square error. `Input`: the gap between the observed values and the predicted values. `Return`: mean square error.
- **`calculate_mae`**: Calculate the mean average error. `Input`: the gap between the observed values and the predicted values. `Return`: mean average error.
- **`compute_loss`**: Calculate the MSE using observed values, features, and models. `Input`: labels, features, weights. `Return` mean square error.
- **`build_poly_feature`**: build the polynomial features for degree = 3. `Input`: features. `Return`: polynomial features.
- **`calculate_combination`**: calculate the number of results when choosing y items from x items. `Input`: the number of total items, the number of items people want to pick. `Return`: the number of all possible combinations.

### `implementations.py`
Contain the mandatory implementations of  6 regression models for this project

**Algorithms for Regression**
- **`least_squares_GD`**: Linear regression using gradient descent. `Input`: labels, features, initial weights, the maximum number of iterations, the step size. `Return`: the final error, the model(the weights).
- **`stochastic_gradient_descent`**: Linear regression using stochastic gradient descent. `Input`: labels, features, initial weights, batch size, the maximum number of iterations, the step size. `Return`: the final error, the model(the weights).
- **`least_squares`**: Least squares regression using normal equations. `Input`: labels, features. `Return`: weights.
- **`ridge_regression`**: Ridge regression using normal equations. `Input`: labels, features, regularisation parameter. `Return`: weights.
- **`ridge_regression_cv`**: Ridge regression using normal equations with cross validations. `Input`: labels, features, regularisation parameter, CV indices, the number of fold. `Return`: weights, correct rate.
- **`logistic_regression`**: Logistic regression using gradient descent or Newton method. `Input`: labels, features, the maximum number of iterations. `Return`: loss, weight.
- **`logistic_regression_penalized_gradient`**: Regularized logistic regression. `Input`: labels, features, the maximum number of iterations. `Return`: loss, weight.

### `run.py`
Contains full process of feature processing, model construction, and prediction. It generates the exact same CSV file submitted on Kaggle.



# EPFL - Machine Learning Project 2

**`Team`:** HuoGuo_Fondue

**`Team Members`:** FENG Wentao, WANG Yunbei, ZHUANG Ying

**Instructions**:
1. Please make sure that the newest `Numpy`,`Pandas`,`sklearn`,`matplotlib` are installed. 
2. Download `train.csv` and `test.csv` from [Kaggle competition](https://www.kaggle.com/c/epfml18-higgs/data), and put them in the same folder as `run.py`
3. Run `run.py`



The followings are some descriptions for the modules or functions we used or designed for this project.

## Defined functions

- **`pooling`**: `Input`:features, power of the data for pooling method(defalut:4). `Return`:the features after pooling. 
- **`CV_Generator`**: Generate grouped cross-validation sets. `Input`: labels, features, labels grouped by video, number of iteration for re-shuffing(defalut:8),proporation of dataset to be put into test set(defalut:0.2). `Return`: a list of train index, a list of test index.
- **`accuracy`**:calculate the accuracy of the model.`Input`: true values of labels, predicted values of labels given by model.`Return`: relative accuracy over the dataset.
- **`rmse`**: calculate the root mean square error.`input`:true values of labels, predicted values of labels given by model. `Return`: root mean square error.
- **`lcc`**: calculate the linear correlation coefficient .`input`:true values of labels, predicted values of labels given by model. `Return`: linear correlation coefficient .
- **`srocc`**: calculate the Spearman's rank correlation coefficient.`input`:true values of labels, predicted values of labels given by model. `Return`: Spearman's rank correlation coefficient.

## External modules

- **`PCA`**:do linear dimensionality reduction to the data features.`input`: the value of the final dimension after reduction.`Return`: the principal component analysis object
- **`GridSearchCv`**:Exhaustive search over selected parameter values for an estimator.`Input`:estimator object,dictationary of parameter names as keys and lists of parameter settings,the grouped cross-validation defined before,using all processors during searching,set to getting more messages during searching,including training scores in the CV_results attribute,assign NAN to error scores,a list of strings to evaluate the predictions on the test set,refit the estimator using the best found root mean square error on the whole dataset,return the average score across folds. `Return`:the grid search with grouped cross-validation object.
- **`SVR`**: support vector regression. `Input`: the kernel type to be used in the algorithm, degree of the polynomial kernel function,the standard deviation of the Gaussian function(gamma),tolerance for stopping criterion,penalty parameter for the error term(C),limitation for iteration for the function(defalut:3000).`Return`:support vector regression object.
- **RadomForestRegressor**:random forest regression. 'Input':seed used by the random number generator(default:8),using all processors when executing the function,the number of trees in the forest,the function used to measure the quality of a split,the depth of the tree,the minimum number of samples required to split the internal node,the minimum number of samples required to be at the leaf node,the number of features to consider when finding the best split,whether using or not the bootstrap samples during the model construction,whether getting or not the message when executing the function. `Return`:random forest regression object.  





