We need to implement some metrics. 
Metrics are simply just `methods` that take some `inputs` and `returns` a `real number`.
> In this case they take the `Ground Observations` and the `Predicted Observations` (I the `Predicted Observations` are for the same parameters as the `Ground Observations`) 

We only need this for #classification and #continuous
# Implementation
---
## Continuous
---
Get some of that good stuff from [here: Performance Metrics in Machine Learning](https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide)
### Mean Squared Error (MSE)
---
- #Regression
```python 
def meanSquaredError(ground: np.ndarray, predictions: np.ndarray) -> float:
	return np.square(np.subtract(ground, predictions)).mean() 
```
The MSE measures the average of error squares i.e. the average squared difference between the estimated values and true values. According to [GFG](https://www.geeksforgeeks.org/python-mean-squared-error/):

1. We need to find the Regression line:   
$$\hat{Y} = \beta_0 + \beta_1 X_i + \varepsilon_i$$
2. Get respective $\hat{Y}$ values for $X$ values by plugging $X$ values into the equation 
$$\hat{Y}$$
3. Now subtract the new value $\hat{Y}$ value from all the original $Y$ values. These values are your `error terms`.  
$$Y_i - \hat{Y_i}$$
4. Square the `error terms`
$$(\hat{Y_i} - Y_i)^2$$
5. Sum up all the `squared errors` 
$$\sum_{i=1}^{N} (\hat{Y_i} - Y_i)^2$$
6. Divide that value by the total number of observations
$$MSE = \frac{1}{N} \sum_{i=1}^{N} (\hat{Y_i} - Y_i)^2$$


Using Code, it looks like this:
```python
import numpy as np 
  
# Given values 
Y_true = [1,1,2,2,4]  # Y_true = Y (original values) 
  
# Calculated values 
Y_pred = [0.6,1.29,1.99,2.69,3.4]  # Y_pred = Y' 
  
# Mean Squared Error 
MSE = np.square(np.subtract(Y_true,Y_pred)).mean() 
```



### Mean Absolute Error (MAE)
---
- `MAE` is measured as the average sum of absolute difference between predictions and actual observations
- `MAE` is more robust against outliers because it does not make use of squares

$$MAE = \frac{1}{N}\sum_{i=1}^{N}|\hat{Y_i}-Y_i|$$
```python
def meanSquaredError(ground: np.ndarray, predictions: np.ndarray) -> float:
	return (np.sum(np.abs(np.subtract(ground,predictions))) / len(ground))
```

### Root Mean Squared Error (RMSE)
---

### R² (R-Squared)
---

## Classification
---
> Getting information from this [EvidentlyAI](https://www.evidentlyai.com/classification-metrics/multi-class-metrics) page. 
> Or this page [Accuracy vs. precision vs. recall in machine learning: what's the difference?](https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall) 

- **Accuracy** shows how often a classification ML model is correct **overall**. 
- **Precision** shows how often an ML model is correct when **predicting the target class.**
- **Recall** shows whether an ML model can find **all objects** **of the target class**. 
- Consider the class balance and costs of different errors when choosing the suitable metric.
### Accuracy
---
> **Accuracy** is a metric that measures how often a machine learning model correctly predicts the outcome. You can calculate accuracy by dividing the number of correct predictions by the total number of predictions.

$$Accuracy=\frac{Correct\_Predictions}{All\_Predictions}$$
- $I$ Below is the indicator function that checks whether the prediction $Y_{(i)}$ is the same as the ground truth $Y$. This function returns 1 if they are equal and 0 if not.
- The sum ($\sum$) then sums up the results from the number $n$ checks. 
- The $\frac{1}{n}$ part then makes a percentage from the summed up values. 
$$Accuracy = \frac{1}{N}\sum_{i=1}^{N}I[\hat{Y_i} = Y_i]$$
Some python-pseudocode would look like this:
```python

def accuracy(ground: np.ndarray, predictions: np.ndarray) -> float:
	values = []
	for gnd, pre in zip(ground, predictions):
		values.append(indicator(gnd, pre))
	return sum(values)/len(values)

def indicator(ground: float|Any?, prediction: float|Any?) -> bool:
	return 1 if ground.equal(prediction) else 0
```

#### Pros and Cons

Let’s sum up the accuracy metric!

**Pros:**
- Accuracy is a helpful metric when you deal with **balanced classes** and care about the overall model “correctness” and not the ability to predict a specific class. 
- Accuracy is **easy to explain** and communicate. 

**Cons:**
- If you have **imbalanced classes**, accuracy is less useful since it gives equal weight to the model’s ability to predict all categories.
- Communicating accuracy in such cases **can be misleading** and disguise low performance on the target class.
### Precision
---
>**Precision** is a metric that measures how often a machine learning model correctly predicts the positive class. You can calculate precision by dividing the number of correct positive predictions (true positives) by the total number of instances the model predicted as positive (both true and false positives).
$$Precision=\frac{True\_Positives}{All\_Positives}$$
$$All\_Positives=True\_Positives+False\_Positives$$
```python
def precision(ground: np.ndarray, predictions: np.ndarray) -> float:
	  
```
#### Pros and Cons
---
**Pros**
- It works well for problems with **imbalanced classes** since it shows the model correctness in identifying the target class.
- Precision is useful when the **cost of a false positive** is high. In this case, you typically want to be **confident in identifying the target class,** even if you miss out on some (or many) instances.

**Cons**
- Precision does not consider **false negatives.** Meaning: it does not account for the cases when we miss our target event!
### Recall
---
> Recall is a metric that measures how often a machine learning model correctly identifies positive instances (true positives) from all the actual positive samples in the dataset. You can calculate recall by dividing the number of true positives by the number of positive instances. The latter includes true positives (successfully identified cases) and false negative results (missed cases).

In other words, recall answers the question: can an ML model find all instances of the positive class?
$$Recall=\frac{True\_Positives}{True\_Positives+False\_Negatives}$$
#### Pros and Cons
---
**Pros**
- It works well for problems with **imbalanced classes** since it is focused on the model’s ability to find objects of the target class.
- Recall is useful when the **cost of false negatives** is high. In this case, you typically want to find **all objects of the target class,** even if this results in some false positives (predicting a positive when it is actually negative).

**Cons**
- Recall is that it does not account for the cost of these **false positives**.

# General Information about `__call__`
### How it works 

So I found some really helpful use case info on a [Real Python](https://realpython.com/python-callable-instances/#exploring-advanced-use-cases-of-__call__) page about the `__call__` method. 
They [[implement]] it using a #DesignPattern called the [[Strategy Pattern]]
[[Showandtell]]

```python
# serializing.py

import json

import yaml

class JsonSerializer:
    def __call__(self, data):
        return json.dumps(data, indent=4)

class YamlSerializer:
    def __call__(self, data):
        return yaml.dump(data)

class DataSerializer:
    def __init__(self, serializing_strategy):
        self.serializing_strategy = serializing_strategy

    def serialize(self, data):
        return self.serializing_strategy(data)
```

This is how you use it:
```python
>>> from serializing import DataSerializer, JsonSerializer, YamlSerializer

>>> data = {
...     "name": "Jane Doe",
...     "age": 30,
...     "city": "Salt Lake City",
...     "job": "Python Developer",
... }

>>> serializer = DataSerializer(JsonSerializer())
>>> print(f"JSON:\n{serializer.serialize(data)}")
JSON:
{
    "name": "Jane Doe",
    "age": 30,
    "city": "Salt Lake City",
    "job": "Python Developer"
}

>>> # Switch strategy
>>> serializer.serializing_strategy = YamlSerializer()
>>> print(f"YAML:\n{serializer.serialize(data)}")
YAML:
age: 30
city: Salt Lake City
job: Python Developer
name: Jane Doe
```
### How you use it
Whenever we want to run some metrics on a model, we will first get an instance of the type of Metric we need.

```python
meanSquaredError = get_metric("mean_squared_error")
```
That will return an instance of the `class MeanSquaredError(Metric)`

We can then proceed to... not finish this... 
### But WTF does the `__call__` function do???

#### Explanation

I started at this [GeeksForGeeks](https://www.geeksforgeeks.org/__call__-in-python/) page about the `__call__` built-in method.

> The `__call__` method allows you to call classes like functions:

```python
x(arg1, arg2, ...)
```

The above line is shorthand for:

```python
x.__call__(arg1, arg2, ...)
```

#### Here is an example from the website:

```python
class Example:
    def __init__(self):
        print("Instance Created")
    
    # Defining __call__ method
    def __call__(self):
        print("Instance is called via special method")

# Instance created
e = Example()

# __call__ method will be called
e()
```

##### Output:

```
Instance Created
Instance is called via special method
```


#### Callable class

```python
class MyCallableClass:
    def __call__(self, *args, **kwargs):
        print(f"Called with arguments: {args} and keyword arguments: {kwargs}")

obj = MyCallableClass()
obj(1, 2, 3, a=4, b=5)
# Output: Called with arguments: (1, 2, 3) and keyword arguments: {'a': 4, 'b': 5}
```
