### Purpose: 
The Mixin pattern is used to address the diamond problem in multiple inheritance.

### Definition: 
A Mixin class provides specific functionality that can be added to multiple classes. Mixins avoid ambiguities in inheritance by keeping inheritance hierarchies simple.

### Example:
```python
class Contact:
    all_contacts = []
    def __init__(self, name, email):
        self.name = name
        self.email = email
        Contact.all_contacts.append(self)

class Sellable:
    def sell(self, item):
        print(f"Sell {item} to {self.name}")

class Customer(Contact, Sellable):
    pass
    ```

### Applications
- Helps modularize functionality, allowing combinations without complex inheritance.
