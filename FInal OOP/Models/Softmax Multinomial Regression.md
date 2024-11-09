# Softmax Multinomial Logistic Regression

## Overview
Softmax regression, also known as multinomial logistic regression, is an extension of logistic regression to handle multiclass classification problems. In softmax regression, we use the **softmax function** to assign probabilities to each class, allowing us to classify inputs into one of multiple categories. Softmax regression is commonly used in machine learning tasks where there are three or more classes to predict.

## Math and Model Explanation

### 1. Hypothesis Function
The softmax function generalizes the sigmoid function used in binary logistic regression to multiple classes. For each class \( k \), we compute a score $( z_k )$ as a linear combination of the input features and model parameters:

$$z_k = X \cdot \theta_k$$

where:
- ( X ) is the input feature matrix (with shape $( m \times n )$, where \( m \) is the number of samples and \( n \) is the number of features).
- $( \theta_k )$ is the weight vector for class ( k ) (with shape $( n \times 1 )$).

The **softmax function** then computes the probability that a given input belongs to class \( k \) by normalizing these scores across all classes \( K \):

$P(y = k | X) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$
### 2. Cost Function (Cross-Entropy Loss)
To evaluate the model's performance, we use the cross-entropy loss function, which penalizes the model for incorrect predictions. The cross-entropy loss for softmax regression is defined as:

$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(P(y = k | X^{(i)}))$

where:
- $( y_k^{(i)} )$ is a binary indicator that is 1 if sample \( i \) belongs to class \( k \), and 0 otherwise.
- $( P(y = k | X^{(i)}) )$ is the predicted probability of class \( k \) for sample \( i \).

### 3. Gradient Descent for Optimization
To minimize the cross-entropy loss, we use gradient descent. The update rule for each weight \( \theta_j \) in class \( k \) is:

$\theta_{j}^{(k)} := \theta_{j}^{(k)} - \alpha \frac{\partial J(\theta)}{\partial \theta_{j}^{(k)}}$

where $( \alpha )$ is the learning rate.

The gradient of the cost function with respect to each parameter $( \theta_{j}^{(k)} )$ is:

$\frac{\partial J(\theta)}{\partial \theta_{j}^{(k)}} = \frac{1}{m} \sum_{i=1}^{m} \left( P(y = k | X^{(i)}) - y_k^{(i)} \right) X_j^{(i)}$
## Code Implementation
Here’s the code for a Softmax Regression model implemented using `numpy`. This implementation includes training using gradient descent, predicting probabilities, and making final class predictions.

```python
import numpy as np

class SoftmaxRegressionModel:
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000) -> None:
        """
        Initializes the softmax regression model with given hyperparameters.
        
        Parameters:
        - learning_rate: float - The step size for gradient descent.
        - num_iterations: int - The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the softmax of each row of the input z.
        
        Parameters:
        - z: np.ndarray - The linear model output (logits) for each class.
        
        Returns:
        - np.ndarray - The softmax probabilities.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the cross-entropy cost function for softmax regression.
        
        Parameters:
        - X: np.ndarray - The feature matrix.
        - y: np.ndarray - The actual class labels.
        
        Returns:
        - float - The cost.
        """
        m = X.shape[0]
        h = self.softmax(X.dot(self.theta))
        y_one_hot = np.eye(self.theta.shape[1])[y]
        cost = -(1 / m) * np.sum(y_one_hot * np.log(h))
        return cost

    def gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Perform gradient descent to minimize the cost function.
        
        Parameters:
        - X: np.ndarray - The feature matrix.
        - y: np.ndarray - The actual class labels.
        """
        m = X.shape[0]
        y_one_hot = np.eye(self.theta.shape[1])[y]  # Convert labels to one-hot encoding

        for _ in range(self.num_iterations):
            h = self.softmax(X.dot(self.theta))
            gradient = (1 / m) * X.T.dot(h - y_one_hot)
            self.theta -= self.learning_rate * gradient

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the training data.
        
        Parameters:
        - X: np.ndarray - The feature matrix.
        - y: np.ndarray - The target labels.
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
        self.theta = np.zeros((X.shape[1], len(np.unique(y))))  # Initialize theta
        self.gradient_descent(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input data.
        
        Parameters:
        - X: np.ndarray - The feature matrix.
        
        Returns:
        - np.ndarray - The predicted probabilities for each class.
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return self.softmax(X.dot(self.theta))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the input data.
        
        Parameters:
        - X: np.ndarray - The feature matrix.
        
        Returns:
        - np.ndarray - The predicted class labels.
        """
        return np.argmax(self.predict_proba(X), axis=1)
```

## Example Usage

Here's how to use the `SoftmaxRegressionModel` class for multiclass classification:

```python
# Sample training data
X_train = np.array([[0.5, 1.5], [1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [3.0, 3.5]])
y_train = np.array([0, 1, 2, 1, 0])  # Multiclass labels

# Initialize and train the model
model = SoftmaxRegressionModel(learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([[1.0, 2.0], [3.0, 3.5]])
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```
## Pros and Cons

### Pros

- **Handles multiclass classification** naturally by computing probabilities for each class.
- **Probabilistic interpretation**: Returns probabilities for each class, allowing for confidence estimation.
- **Good for linearly separable data** and can work well with high-dimensional data.

### Cons

- **Assumes linear decision boundaries**: May not perform well on data with complex, non-linear relationships between classes.
- **Sensitive to outliers**: Like other linear models, outliers can impact the decision boundaries.
- **Computationally expensive for large datasets**: The need to compute scores for each class can slow down training on large datasets with many classes.