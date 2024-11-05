We need to implement some metrics. 
Metrics are simply just `methods` that take some `inputs` and `returns` a `real number`.

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
