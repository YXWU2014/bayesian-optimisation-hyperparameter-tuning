# Data Analytics and Predictive Modeling for Boston Housing Prices



### Why

This project is aimed at predicting house prices in the Boston area. With the rise in demand for houses and the subsequent increase in house prices, it has become vital to predict these prices accurately for both buyers and sellers. This project uses a dataset of Boston house prices, as well as several potential influencing factors (or features), to build a predictive model.



### Data Source

The data used in this project comes from the Boston Housing dataset, available within the `Keras` library in Python. This dataset contains 506 observations (split into training and testing data), with 13 features including crime rate, the average number of rooms per dwelling, accessibility to radial highways, and others.  

### Goal

The goal of this project is to build a model capable of predicting house prices in Boston with high accuracy. To achieve this, we train a neural network model using TensorFlow and Keras, and then optimise the hyperparameters using Bayesian Optimization.

### How It's Been Done

1. **Data Preparation:** The Boston Housing data is loaded, and the features are scaled using StandardScaler from the sklearn library.
2. **Model Building:** We use TensorFlow and Keras to build a deep neural network. This model includes hidden layers with the 'ReLU' activation function, a dropout layer for regularisation, and an output layer with a linear activation function. The loss function can be either Mean Squared Error (MSE) or Mean Absolute Error (MAE), determined by the parameter `loss_class`.
3. **Model Training and Evaluation:** The model is trained on the scaled training data and evaluated on the test data. The evaluation metrics used are the chosen loss function (MSE or MAE) and the R^2 score.
4. **Hyperparameter Optimization:** We use Bayesian Optimization (GPyOpt library) to optimise the hyperparameters of the neural network model, namely the number of hidden layers, the number of nodes, the dropout rate, and the loss function.
5. **Ensemble Learning:** We run the Bayesian Optimization multiple times (an ensemble) to obtain a range of optimal hyperparameters. The performance of the ensemble is then evaluated and visualised.

The entire process is run using Python and makes use of libraries such as numpy, pandas, Tensorflow, Keras, sklearn, GPyOpt, joblib, seaborn, and matplotlib.

### Impact

The resulting model can be used by real estate companies or individuals to predict the price of houses in the Boston area, given a set of features. This can assist in decision-making processes related to buying or selling houses. Additionally, the use of Bayesian Optimization demonstrates a method for tuning neural network hyperparameters that can generalise to other data sets and model types.