### Purpose: 
The Decorator pattern adds additional functionality to methods or functions dynamically without modifying their structure.

### Definition: 
In Python, decorators wrap a function or method to extend its behavior.

**Example**:
```python
from functools import wraps

def log_args(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with {args} and {kwargs}")
        return func(*args, **kwargs)
    return wrapper

@log_args
def add(x, y):
    return x + y

# Output: Calling add with (1, 2) and {}
```

### Application
- Used to add logging, access control, or timing functionality to methods or functions without altering their code.