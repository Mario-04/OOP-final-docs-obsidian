### Purpose: 
Simplifies a complex system by providing a unified interface, often wrapping an existing system to streamline interactions.

### Definition: 
A façade class provides specific functionality by restricting an interface, often through another class’s methods.

**Example**:
```python
from sklearn.linear_model import Lasso

class LassoFacade:
    def __init__(self, **kwargs):
        self._model = Lasso(**kwargs)
    
    def fit(self, X, y):
        self._model.fit(X, y)
    
    def predict(self, X):
        return self._model.predict(X)
```


### Applications
- Wraps complex or generic classes to limit functionality and enforce an interface.
- Useful in ML to standardize model APIs.
