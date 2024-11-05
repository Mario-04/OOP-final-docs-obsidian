### Purpose: 
Allows sequential access to elements in a container without exposing the underlying structure.

### Definition: 
An iterator provides methods to traverse a collection.

**Example**:
```python
class FileReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path) as file:
            for line in file:
                yield line.strip()
```

### Application
- Used for data structures that need custom iteration logic.