We need to implement some metrics. 
Metrics are simply just `methods` that take some `inputs` and `returns` a `real number`.
# Implementation
## Mean Squared Error (MSE)
- #Regression
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

## Accuracy
- #Regression
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

## Mean Absolute Error (MAE)
- `MAE` is measured as the average sum of absolute difference between predictions and actual observations
- `MAE` is more robust against outliers because it does not make use of squares

$$MAE = \frac{1}{N}\sum_{i=1}^{N}|\hat{Y_i}-Y_i|$$
```python
def meanSquaredError(ground: np.ndarray, predictions: np.ndarray) -> float:

```

# General Information
### How it works 

So I found some really helpful use case info on a [Real Python](https://realpython.com/python-callable-instances/#exploring-advanced-use-cases-of-__call__) page about the `__call__` method. 
They implement it using a #DesignPattern called the [[Strategy Pattern]]

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
