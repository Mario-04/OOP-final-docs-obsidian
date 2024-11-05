### Purpose: 
Ensures that a class has only one instance and provides a global point of access to that instance.

### Definition: 
A singleton class restricts instantiation to a single object.

**Example**:
```python
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class ContactList(metaclass=SingletonMeta):
    pass

cl1 = ContactList()
cl2 = ContactList()
assert cl1 is cl2  # True
```

### Applications
- Useful for shared resources like database connections or configurations where only one instance is needed.