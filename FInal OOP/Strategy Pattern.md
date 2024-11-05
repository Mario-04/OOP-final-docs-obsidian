### Purpose: 
Encapsulates different algorithms or strategies within classes, allowing them to be selected at runtime based on the context.

### Definition: A context class refers to a strategy interface to execute a behavior, allowing interchangeable strategies.

**Example**:
```python
class RandomKaiming:
    def initialize(self, parameters):
        # initialization logic

class NeuralNetwork:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def initialize(self):
        self.strategy.initialize(self.parameters)

# Usage
network = NeuralNetwork(strategy=RandomKaiming())
network.initialize()
```

### Applications
- Useful when there are multiple ways to perform an action, and the best approach is chosen based on context.