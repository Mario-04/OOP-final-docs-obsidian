In model, we need to implement a 1. Regression model, 2. Classification model
> The Models that inherit from this class:
> [[Linear Regression]]
> [[Logistic Regression]]
> [[KNearestNeighbour]]
> [[Polynomial Regression]]

New information from chatgpt
> [[Softmax Multinomial Regression]]
> [[Decision Tree Classifier]]
> 

Both are known as supervised learning algorithms and work with **labelled** data sets

  

**Model**: a function that maps input features to a target feature derived from a set of observations. Can represent either a classification or a regression task, as seen during lectures and assignments. Models have a fit behavior that allows them to be trained on a given task and a predict behavior that allows them to make predictions on the same task.

  

# Implementation of Regression models

### There are multiple different types of regression models 

Regression finds correlations between dependent and independent variables.

###### Example Assignment 1:

- Multiple linear regression model

- A multiple linear regression model indicates that there is more than one input variable may affect the outcome or target variable.

- Linear regression model 

- Where the relationship between inputs and outputs is a straight line.

- In addition, the model fit can be described using a mean squared error. This basically gives us a number to show exactly how well the linear model fits.

![[Screenshot 2024-11-08 at 14.40.27.png]]

  

# Implementation of classification models

Classification is an algorithm that finds functions that help divide the dataset into classes based on various parameters

### A computer program gets taught on the training dataset and categorises the data into various categories depending on what it learned.

  

![[Screenshot 2024-11-08 at 14.40.06.png]]

  

  

# So now we know what a regression model and classification model is!

  

# Implementation in final project

In **autoop/core/ml/model/__init__.py** 

- REGRESSION_MODELS = [

] # add your models as str here

  

- CLASSIFICATION_MODELS = [

] # add your models as str here

  

**Here we input the models we created so that the user can choose an option to do.**

  

- def get_model(model_name: str) -> Model:

"""Factory function to get a model by name."""

raise NotImplementedError("To be implemented.")

  

- **This is our call function to go and get the method that the user wants to do** 

  

### We call either the classification folder or the regression folder

- autoop/core/ml/model/classification/__init__.py

- autoop/core/ml/model/regression/__init__.py

  

Examples of how each model we are going to implement are going to be called: 

![[Screenshot 2024-11-08 at 14.46.30.png]]

  

So we would implement a regression model of multiple_linear_regression that can be used here:

We need 3 for both regression and classification

  

#### Classification 

- K nearest neighbour **KNN**

  

  

#### Regression 

- multiple linear regression