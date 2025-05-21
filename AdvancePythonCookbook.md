# Advanced Python Cookbook for Production Development

## Table of Contents
1. [Basic Concepts](#basic-concepts)
   - [Python Environment Setup](#python-environment-setup)
   - [Virtual Environments](#virtual-environments)
   - [Package Management](#package-management)
   - [Code Organization](#code-organization)
   - [Type Hints](#type-hints)

2. [Intermediate Concepts](#intermediate-concepts)
   - [Object-Oriented Programming](#object-oriented-programming)
   - [Functional Programming](#functional-programming)
   - [Error Handling and Exceptions](#error-handling-and-exceptions)
   - [Context Managers](#context-managers)
   - [Generators and Iterators](#generators-and-iterators)
   - [Decorators](#decorators)
   - [Concurrency Basics](#concurrency-basics)

3. [Advanced Concepts](#advanced-concepts)
   - [Metaprogramming](#metaprogramming)
   - [Advanced Concurrency](#advanced-concurrency)
   - [Performance Optimization](#performance-optimization)
   - [Memory Management](#memory-management)
   - [Design Patterns](#design-patterns)
   - [Testing Strategies](#testing-strategies)
   - [Packaging and Distribution](#packaging-and-distribution)

4. [Enterprise-Level Practices](#enterprise-level-practices)
   - [Application Architecture](#application-architecture)
   - [API Development](#api-development)
   - [Database Interaction](#database-interaction)
   - [Logging and Monitoring](#logging-and-monitoring)
   - [Security Best Practices](#security-best-practices)
   - [Configuration Management](#configuration-management)
   - [Continuous Integration/Continuous Deployment](#continuous-integrationcontinuous-deployment)
   - [Containerization and Orchestration](#containerization-and-orchestration)

---

## Basic Concepts

### Python Environment Setup

#### Python Versions and Installation

While you're already familiar with Python, ensuring you have the right setup is crucial for production development:

```python
# Check Python version
import sys
print(f"Python version: {sys.version}")
print(f"Python version info: {sys.version_info}")
```

For production applications, Python 3.10+ is recommended due to performance improvements and important language features.

#### Managing Multiple Python Versions with pyenv

```bash
# Install pyenv (Linux/Mac)
curl https://pyenv.run | bash

# Install specific Python version
pyenv install 3.11.2

# Set global Python version
pyenv global 3.11.2

# Set local Python version for a project
cd your_project
pyenv local 3.10.9
```

### Virtual Environments

Virtual environments are essential for isolating project dependencies and ensuring reproducible environments.

#### Using venv (built-in)

```bash
# Create a virtual environment
python -m venv myenv

# Activate the virtual environment
# On Windows
myenv\Scripts\activate
# On Unix or MacOS
source myenv/bin/activate

# Deactivate the virtual environment
deactivate
```

#### Using Poetry (modern dependency management)

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Create a new project
poetry new my-project

# Initialize Poetry in an existing project
cd existing-project
poetry init

# Add dependencies
poetry add requests

# Add development dependencies
poetry add --dev pytest

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Update dependencies
poetry update
```

### Package Management

#### Requirements Files

```bash
# Generate requirements.txt
pip freeze > requirements.txt

# Install from requirements.txt
pip install -r requirements.txt
```

#### Using Poetry for Dependency Management

Poetry provides more sophisticated dependency resolution than pip:

```toml
# pyproject.toml
[tool.poetry]
name = "my-project"
version = "0.1.0"
description = "My Python project"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
requests = "^2.28.1"

[tool.poetry.group.dev.dependencies]
pytest = "^7.2.0"
black = "^23.1.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

#### Version Pinning Strategies

```toml
# Fixed versions (exact match)
requests = "2.28.1"

# Compatible releases (SemVer)
requests = "^2.28.1"  # >= 2.28.1, < 3.0.0

# Greater than or equal to
requests = ">=2.28.1"

# Range specification
requests = ">=2.28.1,<2.30.0"
```

### Code Organization

#### Project Structure

A well-organized Python project structure:

```
my_project/
│
├── pyproject.toml         # Project metadata and dependencies
├── README.md              # Project documentation
├── LICENSE                # License information
├── .gitignore             # Git ignore file
│
├── src/                   # Source code
│   └── my_project/        # Main package
│       ├── __init__.py    # Package initialization
│       ├── module1.py     # Module 1
│       ├── module2.py     # Module 2
│       └── subpackage/    # Subpackage
│           ├── __init__.py
│           └── module3.py
│
├── tests/                 # Test directory
│   ├── __init__.py
│   ├── test_module1.py
│   └── test_module2.py
│
├── docs/                  # Documentation
│
└── scripts/               # Utility scripts
```

#### Modules and Packages

```python
# Creating a package
# __init__.py
"""My awesome package."""

from .module1 import function1
from .module2 import Class1

__all__ = ['function1', 'Class1']
```

#### Import Best Practices

```python
# Absolute imports (preferred)
from my_project.subpackage import module3
from my_project.module1 import function1

# Relative imports (use with caution)
from ..module1 import function1  # Go up one directory level
from . import module3            # From the same directory
```

### Type Hints

Type hints improve code readability, enable static analysis, and help catch bugs early:

```python
# Basic type hints
def calculate_total(price: float, quantity: int) -> float:
    return price * quantity

# Complex type hints
from typing import List, Dict, Optional, Union, Callable, TypeVar, Generic

# For lists
def process_items(items: List[str]) -> List[int]:
    return [len(item) for item in items]

# For dictionaries
def process_data(data: Dict[str, int]) -> Dict[str, float]:
    return {key: value / 2 for key, value in data.items()}

# Optional parameters
def fetch_user(user_id: int, include_details: Optional[bool] = None) -> Dict[str, Union[str, int]]:
    result = {"id": user_id, "name": "John Doe"}
    if include_details:
        result["details"] = "Some details"
    return result

# Type aliases
UserId = int
UserData = Dict[str, Union[str, int, bool]]

def get_user_info(user_id: UserId) -> UserData:
    return {"id": user_id, "name": "Jane Doe", "active": True}

# Generic types
T = TypeVar('T')

def first_element(collection: List[T]) -> Optional[T]:
    return collection[0] if collection else None
```

#### Type Checking with mypy

```bash
# Install mypy
pip install mypy

# Run mypy on your code
mypy my_project/
```

```python
# mypy.ini configuration
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True

[mypy.plugins.numpy.dynamo]
enable = True
```

## Intermediate Concepts

### Object-Oriented Programming

#### Class Design

```python
# Basic class structure
class Customer:
    """A customer representation."""
    
    def __init__(self, customer_id: int, name: str) -> None:
        self.customer_id = customer_id
        self.name = name
        self._purchases: List[Purchase] = []
    
    def add_purchase(self, purchase: 'Purchase') -> None:
        self._purchases.append(purchase)
    
    @property
    def total_spent(self) -> float:
        return sum(purchase.amount for purchase in self._purchases)
    
    def __str__(self) -> str:
        return f"Customer {self.name} (ID: {self.customer_id})"
    
    def __repr__(self) -> str:
        return f"Customer(customer_id={self.customer_id}, name='{self.name}')"
```

#### Inheritance and Composition

```python
# Inheritance
class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age
    
    def display_info(self) -> str:
        return f"{self.name}, {self.age} years old"

class Employee(Person):
    def __init__(self, name: str, age: int, employee_id: str) -> None:
        super().__init__(name, age)
        self.employee_id = employee_id
    
    def display_info(self) -> str:
        return f"{super().display_info()}, ID: {self.employee_id}"

# Composition (often preferred over inheritance)
class Address:
    def __init__(self, street: str, city: str, state: str, zip_code: str) -> None:
        self.street = street
        self.city = city
        self.state = state
        self.zip_code = zip_code

class Customer:
    def __init__(self, name: str, address: Address) -> None:
        self.name = name
        self.address = address  # Composition
```

#### Abstract Base Classes and Interfaces

```python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    @abstractmethod
    def process_payment(self, amount: float) -> bool:
        """Process a payment."""
        pass
    
    @abstractmethod
    def refund(self, amount: float) -> bool:
        """Process a refund."""
        pass

class CreditCardProcessor(PaymentProcessor):
    def process_payment(self, amount: float) -> bool:
        print(f"Processing credit card payment of ${amount}")
        return True
    
    def refund(self, amount: float) -> bool:
        print(f"Processing credit card refund of ${amount}")
        return True
```

#### Dataclasses

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class Product:
    id: int
    name: str
    price: float
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    
    def discount_price(self, percentage: float) -> float:
        return self.price * (1 - percentage / 100)
```

### Functional Programming

#### Lambda Functions

```python
# Simple lambda
multiply = lambda x, y: x * y
result = multiply(5, 3)  # 15

# Lambda with sorting
people = [{'name': 'John', 'age': 30}, {'name': 'Jane', 'age': 25}]
sorted_people = sorted(people, key=lambda person: person['age'])
```

#### Map, Filter, and Reduce

```python
from functools import reduce

# Map: Apply function to each item
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]

# Filter: Select items based on condition
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]

# Reduce: Aggregate items using a function
sum_all = reduce(lambda acc, val: acc + val, numbers)  # 15

# List comprehensions (often preferred over map/filter)
squared = [x**2 for x in numbers]  # [1, 4, 9, 16, 25]
even_numbers = [x for x in numbers if x % 2 == 0]  # [2, 4]
```

#### Higher-Order Functions

```python
def create_multiplier(factor: int) -> Callable[[int], int]:
    def multiply(x: int) -> int:
        return x * factor
    return multiply

double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15
```

#### Immutability and Pure Functions

```python
# Pure function
def add_pure(x: int, y: int) -> int:
    return x + y

# Impure function
total = 0
def add_impure(x: int) -> int:
    global total
    total += x
    return total

# Working with immutable data
from typing import Dict, Tuple, Any

def update_dict_immutable(data: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    return {**data, key: value}  # Create a new dictionary

original = {"name": "John", "age": 30}
updated = update_dict_immutable(original, "age", 31)
# original = {"name": "John", "age": 30}
# updated = {"name": "John", "age": 31}
```

### Error Handling and Exceptions

#### Try-Except Blocks

```python
def divide(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        print("Error: Division by zero!")
        return float('inf')
    except TypeError as e:
        print(f"Error: {e}")
        raise
    finally:
        print("Division operation attempted")
```

#### Custom Exceptions

```python
class ApplicationError(Exception):
    """Base class for application-specific exceptions."""
    pass

class ResourceNotFoundError(ApplicationError):
    """Raised when a requested resource is not found."""
    def __init__(self, resource_type: str, resource_id: str) -> None:
        self.resource_type = resource_type
        self.resource_id = resource_id
        message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(message)

# Using custom exceptions
def fetch_user(user_id: str) -> Dict[str, Any]:
    if not user_exists(user_id):
        raise ResourceNotFoundError("User", user_id)
    # ...
```

#### Exception Chaining

```python
try:
    # Some operation
    process_data()
except ValueError as e:
    # Chain exceptions to preserve context
    raise ApplicationError("Invalid data format") from e
```

#### Handling Multiple Exceptions

```python
try:
    result = complex_operation()
except (ValueError, TypeError) as e:
    # Handle multiple exception types
    print(f"Operation error: {e}")
except Exception as e:
    # Catch-all for other exceptions
    print(f"Unexpected error: {e}")
    # Re-raise to preserve the error
    raise
```

### Context Managers

#### Using with Statement

```python
# File operations with context manager
with open('file.txt', 'r') as file:
    content = file.read()
# File is automatically closed when the block exits

# Multiple context managers
with open('input.txt', 'r') as input_file, open('output.txt', 'w') as output_file:
    content = input_file.read()
    output_file.write(content.upper())
```

#### Creating Custom Context Managers

```python
# Using a class
class DatabaseConnection:
    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to database: {self.connection_string}")
        self.connection = connect_to_db(self.connection_string)
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        if self.connection:
            self.connection.close()
        # Return True to suppress exceptions, False to propagate them
        return False

# Using contextlib
from contextlib import contextmanager

@contextmanager
def database_connection(connection_string: str):
    connection = None
    try:
        print(f"Connecting to database: {connection_string}")
        connection = connect_to_db(connection_string)
        yield connection
    finally:
        print("Closing database connection")
        if connection:
            connection.close()

# Usage
with database_connection("postgres://user:pass@localhost:5432/db") as conn:
    results = conn.execute("SELECT * FROM users")
```

### Generators and Iterators

#### Creating Generators

```python
def count_up_to(limit: int):
    count = 1
    while count <= limit:
        yield count
        count += 1

# Using a generator
for number in count_up_to(5):
    print(number)  # Prints 1, 2, 3, 4, 5

# Generator expressions
squares = (x * x for x in range(10))
for square in squares:
    print(square)
```

#### Implementing Iterators

```python
class Fibonacci:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.previous = 0
        self.current = 1
        self.count = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count >= self.limit:
            raise StopIteration
        
        self.count += 1
        if self.count == 1:
            return 0
        
        result = self.current
        self.current, self.previous = self.previous + self.current, self.current
        return result

# Using the iterator
for num in Fibonacci(10):
    print(num)  # Prints the first 10 Fibonacci numbers
```

#### Generator Use Cases

```python
def read_large_file(file_path: str, chunk_size: int = 1024):
    """Memory-efficient file reading."""
    with open(file_path, 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Processing a large file without loading it entirely into memory
for chunk in read_large_file('large_file.txt'):
    process_data(chunk)
```

### Decorators

#### Basic Decorators

```python
# Simple decorator
def log_function_call(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

@log_function_call
def add(a, b):
    return a + b

# Equivalent to:
# add = log_function_call(add)

result = add(3, 5)  # Logs the call and returns 8
```

#### Decorators with Parameters

```python
def repeat(times: int):
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(times=3)
def greet(name: str) -> str:
    return f"Hello, {name}!"

print(greet("World"))  # Returns ["Hello, World!", "Hello, World!", "Hello, World!"]
```

#### Class Decorators

```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self, connection_string: str) -> None:
        print(f"Initializing database with {connection_string}")
        self.connection_string = connection_string

# Only one instance will be created
db1 = Database("connection1")
db2 = Database("connection2")  # No initialization happens here
print(db1 is db2)  # True
```

#### Preserving Function Metadata

```python
from functools import wraps

def log_function_call(func):
    @wraps(func)  # Preserves the original function's metadata
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_function_call
def add(a, b):
    """Add two numbers."""
    return a + b

print(add.__name__)  # 'add' (not 'wrapper')
print(add.__doc__)   # 'Add two numbers.' (preserved)
```

### Concurrency Basics

#### Threading

```python
import threading
import time
from typing import List

def worker(name: str, delay: float) -> None:
    print(f"Worker {name} starting")
    time.sleep(delay)
    print(f"Worker {name} finished")

# Create threads
threads: List[threading.Thread] = []
for i in range(5):
    t = threading.Thread(target=worker, args=(f"Thread-{i}", i * 0.5))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

print("All threads completed")
```

#### Thread Safety

```python
import threading

# Thread-safe counter using a lock
class Counter:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self) -> None:
        with self.lock:
            self.value += 1
    
    def get_value(self) -> int:
        with self.lock:
            return self.value

# Thread-local storage
thread_local = threading.local()
thread_local.value = 0

def increment_local():
    thread_local.value += 1
    print(f"Thread {threading.current_thread().name}: {thread_local.value}")
```

#### Multiprocessing

```python
import multiprocessing
import time

def worker(name: str, delay: float) -> None:
    print(f"Process {name} starting")
    time.sleep(delay)
    print(f"Process {name} finished")

if __name__ == "__main__":
    # Create processes
    processes = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(f"Process-{i}", i * 0.5))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All processes completed")
```

#### Concurrent.futures

```python
import concurrent.futures
import time

def worker(name: str) -> str:
    time.sleep(1)
    return f"Worker {name} completed"

# Using ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(worker, f"Thread-{i}"): i for i in range(5)}
    
    for future in concurrent.futures.as_completed(futures):
        worker_id = futures[future]
        try:
            result = future.result()
            print(f"Worker {worker_id} result: {result}")
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")

# Using ProcessPoolExecutor
with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(worker, [f"Process-{i}" for i in range(5)]))
    
    for result in results:
        print(result)
```

## Advanced Concepts

### Metaprogramming

#### Metaclasses

```python
# Metaclass example
class MetaLogger(type):
    def __new__(mcs, name, bases, attrs):
        # Add logging to all methods
        for attr_name, attr_value in attrs.items():
            if callable(attr_value) and not attr_name.startswith("__"):
                attrs[attr_name] = MetaLogger.log_method(attr_value)
        
        return super().__new__(mcs, name, bases, attrs)
    
    @staticmethod
    def log_method(method):
        def wrapper(*args, **kwargs):
            print(f"Calling method {method.__name__}")
            return method(*args, **kwargs)
        return wrapper

# Using the metaclass
class Service(metaclass=MetaLogger):
    def process(self, data):
        return f"Processed: {data}"
    
    def analyze(self, data):
        return f"Analysis result: {data}"

service = Service()
result = service.process("sample data")  # Logs "Calling method process"
```

#### Class Decorators

```python
def add_methods(cls):
    def new_method(self, x):
        return x * 2
    
    cls.new_method = new_method
    return cls

@add_methods
class MyClass:
    def __init__(self, value):
        self.value = value

# Now MyClass has a new_method
obj = MyClass(5)
print(obj.new_method(10))  # 20
```

#### Descriptors

```python
class Validator:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]
    
    def __set__(self, instance, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be at least {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} cannot exceed {self.max_value}")
        instance.__dict__[self.name] = value

class Person:
    age = Validator(min_value=0, max_value=120)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

# This works
person = Person("John", 30)

# This raises ValueError
try:
    person.age = 150
except ValueError as e:
    print(e)  # "age cannot exceed 120"
```

#### Dynamic Attribute Access

```python
class DynamicAttributes:
    def __init__(self):
        self._data = {}
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value
    
    def __delattr__(self, name):
        if name in self._data:
            del self._data[name]
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

# Using dynamic attributes
obj = DynamicAttributes()
obj.name = "John"
obj.age = 30
print(obj.name)  # John
```

### Advanced Concurrency

#### Asyncio Basics

```python
import asyncio

async def async_task(name, delay):
    print(f"Task {name} started")
    await asyncio.sleep(delay)  # Non-blocking sleep
    print(f"Task {name} completed after {delay} seconds")
    return f"Result from {name}"

async def main():
    # Run tasks concurrently
    results = await asyncio.gather(
        async_task("Task 1", 2),
        async_task("Task 2", 1),
        async_task("Task 3", 3)
    )
    
    print("All tasks completed")
    print(f"Results: {results}")

# Run the event loop
asyncio.run(main())
```

#### Async Context Managers

```python
import asyncio

class AsyncDatabase:
    async def __aenter__(self):
        print("Connecting to database")
        await asyncio.sleep(1)  # Simulating connection time
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        await asyncio.sleep(0.5)  # Simulating disconnection time
    
    async def query(self, sql):
        print(f"Executing query: {sql}")
        await asyncio.sleep(0.5)  # Simulating query execution
        return ["result1", "result2"]

async def main():
    async with AsyncDatabase() as db:
        results = await db.query("SELECT * FROM users")
        print(f"Query results: {results}")

asyncio.run(main())
```

#### Async Generators

```python
import asyncio

async def async_range(start, stop):
    for i in range(start, stop):
        await asyncio.sleep(0.1)  # Non-blocking sleep
        yield i

async def main():
    async for i in async_range(1, 5):
        print(f"Received: {i}")

asyncio.run(main())
```

#### Combining Asyncio with Threads and Processes

```python
import asyncio
import concurrent.futures
import time

def cpu_bound_task(n):
    """A CPU-bound task to demonstrate ProcessPoolExecutor."""
    result = 0
    for i in range(n * 10000000):
        result += i
    return result

async def main():
    # For CPU-bound tasks, use ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as process_pool:
        loop = asyncio.get_running_loop()
        
        # Schedule the CPU-bound task in the process pool
        result = await loop.run_in_executor(
            process_pool, cpu_bound_task, 1
        )
        
        print(f"CPU-bound task result: {result}")
    
    # For I/O-bound blocking operations, use ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as thread_pool:
        # Schedule blocking I/O in the thread pool
        result = await loop.run_in_executor(
            thread_pool, time.sleep, 1
        )
        
        print("I/O-bound task completed")

asyncio.run(main())
```

### Performance Optimization

#### Profiling

```python
import cProfile
import pstats
from pstats import SortKey

# Profile a function
def profile_function(func, *args, **kwargs):
    profile = cProfile.Profile()
    profile.enable()
    
    result = func(*args, **kwargs)
    
    profile.disable()
    ps = pstats.Stats(profile).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(10)
    
    return result

# Usage example
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

profile_function(fibonacci, 30)

# Line profiler for detailed line-by-line analysis
# pip install line_profiler
# Usage with IPython: %load_ext line_profiler
# %lprun -f function_name function_name(arguments)
```

#### CPU and Memory Optimization

```python
# Use built-in functions and libraries when possible
# Slow
result = 0
for i in range(1000000):
    result += i

# Fast
result = sum(range(1000000))

# Use appropriate data structures
from collections import defaultdict, Counter

# Instead of:
word_counts = {}
for word in words:
    if word not in word_counts:
        word_counts[word] = 0
    word_counts[word] += 1

# Use:
word_counts = defaultdict(int)
for word in words:
    word_counts[word] += 1

# Or even better:
word_counts = Counter(words)

# Use generators for memory efficiency
# Memory-intensive
all_data = [process(item) for item in large_dataset]
result = sum(all_data)

# Memory-efficient
result = sum(process(item) for item in large_dataset)
```

#### Numba for Performance

```python
# pip install numba
import numpy as np
from numba import jit

# Regular Python function
def python_sum_array(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Numba-optimized function
@jit(nopython=True)
def numba_sum_array(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Comparison
data = np.random.random(10000000)

%time result1 = python_sum_array(data)
%time result2 = numba_sum_array(data)
# Numba typically provides significant performance improvements
```

#### Cython for Performance

```python
# demo.pyx file
def fibonacci_cy(int n):
    cdef int a = 0, b = 1, i, temp
    if n <= 0:
        return a
    for i in range(2, n + 1):
        temp = a + b
        a = b
        b = temp
    return b

# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("demo.pyx")
)

# Build with: python setup.py build_ext --inplace
# Then import and use:
from demo import fibonacci_cy
result = fibonacci_cy(30)
```

### Memory Management

#### Monitoring Memory Usage

```python
import tracemalloc
import linecache
import os

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print(f"Top {limit} lines")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        filename = os.path.basename(frame.filename)
        print(f"#{index}: {filename}:{frame.lineno}: {stat.size / 1024:.1f} KiB")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print(f"    {line}")

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print(f"{len(other)} other: {size / 1024:.1f} KiB")
    total = sum(stat.size for stat in top_stats)
    print(f"Total allocated size: {total / 1024:.1f} KiB")

# Track memory allocations
tracemalloc.start()

# Run your code here
data = [object() for _ in range(1000)]

# Take snapshot and display statistics
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)
```

#### Garbage Collection

```python
import gc

# Force garbage collection
gc.collect()

# Get reference counts
import sys
x = []
print(sys.getrefcount(x))  # Always at least 2 (variable + getrefcount param)

# Inspect garbage collector
print(gc.get_count())  # Get collection counters
print(gc.get_threshold())  # Get collection thresholds

# Disable automatic garbage collection (for performance-critical sections)
gc.disable()
# ... critical code ...
gc.enable()
```

#### Memory Leaks and Circular References

```python
# Circular reference example
class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
    
    def add_child(self, node):
        self.children.append(node)

# Create a circular reference
node1 = Node(1)
node2 = Node(2)
node1.add_child(node2)
node2.add_child(node1)  # Circular reference

# Using weak references to avoid memory leaks
import weakref

class BetterNode:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parent = None
    
    def add_child(self, node):
        self.children.append(node)
        node.parent = weakref.ref(self)  # Weak reference

# Create references without circular memory issues
node1 = BetterNode(1)
node2 = BetterNode(2)
node1.add_child(node2)  # parent is a weak reference
```

### Design Patterns

#### Creational Patterns

```python
# Singleton
class Singleton:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Factory
class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# Builder
class Computer:
    def __init__(self):
        self.cpu = None
        self.memory = None
        self.storage = None
        self.gpu = None

class ComputerBuilder:
    def __init__(self):
        self.computer = Computer()
    
    def with_cpu(self, cpu):
        self.computer.cpu = cpu
        return self
    
    def with_memory(self, memory):
        self.computer.memory = memory
        return self
    
    def with_storage(self, storage):
        self.computer.storage = storage
        return self
    
    def with_gpu(self, gpu):
        self.computer.gpu = gpu
        return self
    
    def build(self):
        return self.computer

# Usage
computer = (ComputerBuilder()
            .with_cpu("Intel i7")
            .with_memory("16GB")
            .with_storage("512GB SSD")
            .with_gpu("NVIDIA RTX 3080")
            .build())
```

#### Structural Patterns

```python
# Adapter
class OldSystem:
    def old_operation(self, data):
        return f"Old operation with {data}"

class NewSystem:
    def new_operation(self, info):
        return f"New operation with {info}"

class SystemAdapter:
    def __init__(self, new_system):
        self.new_system = new_system
    
    def old_operation(self, data):
        return self.new_system.new_operation(data)

# Decorator (OOP implementation)
class Component:
    def operation(self):
        pass

class ConcreteComponent(Component):
    def operation(self):
        return "Base operation"

class Decorator(Component):
    def __init__(self, component):
        self.component = component
    
    def operation(self):
        return self.component.operation()

class LoggingDecorator(Decorator):
    def operation(self):
        print("Logging: Operation started")
        result = self.component.operation()
        print("Logging: Operation completed")
        return result

# Usage
component = ConcreteComponent()
decorated_component = LoggingDecorator(component)
result = decorated_component.operation()
```

#### Behavioral Patterns

```python
# Observer
class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self, message):
        for observer in self._observers:
            observer.update(message)

class Observer:
    def update(self, message):
        pass

class ConcreteObserver(Observer):
    def __init__(self, name):
        self.name = name
    
    def update(self, message):
        print(f"Observer {self.name} received: {message}")

# Strategy
class Strategy:
    def execute(self, data):
        pass

class ConcreteStrategyA(Strategy):
    def execute(self, data):
        return sorted(data)

class ConcreteStrategyB(Strategy):
    def execute(self, data):
        return sorted(data, reverse=True)

class Context:
    def __init__(self, strategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy):
        self.strategy = strategy
    
    def execute_strategy(self, data):
        return self.strategy.execute(data)

# Chain of Responsibility
class Handler:
    def __init__(self):
        self.successor = None
    
    def set_successor(self, successor):
        self.successor = successor
    
    def handle(self, request):
        pass

class ConcreteHandler1(Handler):
    def handle(self, request):
        if request < 10:
            return f"Handler 1 processed request: {request}"
        elif self.successor:
            return self.successor.handle(request)
        return None

class ConcreteHandler2(Handler):
    def handle(self, request):
        if 10 <= request < 20:
            return f"Handler 2 processed request: {request}"
        elif self.successor:
            return self.successor.handle(request)
        return None
```

### Testing Strategies

#### Unit Testing

```python
# test_calculator.py
import unittest
from calculator import Calculator

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def test_add(self):
        self.assertEqual(self.calc.add(1, 2), 3)
        self.assertEqual(self.calc.add(-1, 1), 0)
        self.assertEqual(self.calc.add(-1, -1), -2)
    
    def test_divide(self):
        self.assertEqual(self.calc.divide(10, 2), 5)
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)
    
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
```

#### Pytest

```python
# test_calculator_pytest.py
import pytest
from calculator import Calculator

@pytest.fixture
def calculator():
    return Calculator()

def test_add(calculator):
    assert calculator.add(1, 2) == 3
    assert calculator.add(-1, 1) == 0
    assert calculator.add(-1, -1) == -2

def test_divide(calculator):
    assert calculator.divide(10, 2) == 5
    with pytest.raises(ValueError):
        calculator.divide(10, 0)

# Parameterized tests
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (-1, 1, 0),
    (-1, -1, -2)
])
def test_add_params(calculator, a, b, expected):
    assert calculator.add(a, b) == expected
```

#### Mocking

```python
# Using unittest.mock
from unittest.mock import Mock, patch, MagicMock
import unittest
from user_service import UserService

class TestUserService(unittest.TestCase):
    def test_get_user_details(self):
        # Create a mock database
        mock_db = Mock()
        mock_db.query.return_value = {"id": 1, "name": "John Doe"}
        
        # Inject the mock into the service
        service = UserService(mock_db)
        user = service.get_user_details(1)
        
        # Assertions
        self.assertEqual(user["name"], "John Doe")
        mock_db.query.assert_called_once_with("SELECT * FROM users WHERE id = 1")
    
    @patch('user_service.requests')
    def test_fetch_external_data(self, mock_requests):
        # Setup the mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "example"}
        mock_requests.get.return_value = mock_response
        
        # Create service and call method
        service = UserService(Mock())
        result = service.fetch_external_data("example_api")
        
        # Assertions
        self.assertEqual(result, {"data": "example"})
        mock_requests.get.assert_called_once_with("https://api.example.com/example_api")
```

#### Integration Testing

```python
# Integration test with actual database
import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, User
from user_service import UserService

@pytest.fixture(scope="module")
def db_session():
    # Create test database
    engine = create_engine('sqlite:///test.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Setup test data
    user = User(name="Test User", email="test@example.com")
    session.add(user)
    session.commit()
    
    yield session
    
    # Teardown
    session.close()
    os.remove("test.db")

def test_user_service_integration(db_session):
    service = UserService(db_session)
    user = service.get_user_by_email("test@example.com")
    
    assert user is not None
    assert user.name == "Test User"
```

### Packaging and Distribution

#### Package Structure

```
my_package/
├── pyproject.toml           # Project configuration
├── setup.py                 # Build script
├── setup.cfg                # Configuration for setup.py
├── README.md                # Project readme
├── LICENSE                  # License file
├── CHANGELOG.md             # Change history
├── requirements.txt         # Dependencies
├── requirements-dev.txt     # Development dependencies
├── src/                     # Source code
│   └── my_package/          # Main package directory
│       ├── __init__.py      # Package initialization
│       ├── module1.py       # Module 1
│       └── subpackage/      # Subpackage
│           ├── __init__.py
│           └── module2.py
├── tests/                   # Test directory
│   ├── __init__.py
│   ├── test_module1.py
│   └── test_subpackage/
│       └── test_module2.py
└── docs/                    # Documentation
    ├── conf.py
    ├── index.rst
    └── api.rst
```

#### Modern Packaging with pyproject.toml

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-package"
version = "0.1.0"
description = "A sample Python package"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]
dependencies = [
    "requests>=2.28.0",
    "numpy>=1.23.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.3.0",
    "isort>=5.10.1",
    "mypy>=0.950",
]
docs = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
]

[project.urls]
"Homepage" = "https://github.com/username/my-package"
"Bug Tracker" = "https://github.com/username/my-package/issues"

[project.scripts]
my-command = "my_package.cli:main"

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310"]

[tool.isort]
profile = "black"

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
```

#### Building and Publishing

```bash
# Build the package
python -m build

# Upload to PyPI (requires twine)
python -m pip install twine
python -m twine upload dist/*

# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*
```

## Enterprise-Level Practices

### Application Architecture

#### Layered Architecture

```
my_application/
├── presentation/            # UI/API layer
│   ├── __init__.py
│   ├── api.py
│   └── web.py
├── application/             # Application/Service layer
│   ├── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── user_service.py
│   │   └── product_service.py
│   └── dto/                 # Data Transfer Objects
│       ├── __init__.py
│       └── user_dto.py
├── domain/                  # Domain/Business layer
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── product.py
│   └── repositories/        # Repository interfaces
│       ├── __init__.py
│       ├── user_repository.py
│       └── product_repository.py
└── infrastructure/          # Infrastructure layer
    ├── __init__.py
    ├── config.py
    ├── logging.py
    ├── database/
    │   ├── __init__.py
    │   ├── models.py
    │   └── repositories/    # Repository implementations
    │       ├── __init__.py
    │       ├── user_repository.py
    │       └── product_repository.py
    └── external/
        ├── __init__.py
        └── payment_gateway.py
```

#### Dependency Injection

```python
# Simple dependency injection without a framework
class UserRepository:
    def get_user(self, user_id):
        pass

class DatabaseUserRepository(UserRepository):
    def __init__(self, db_connection):
        self.db_connection = db_connection
    
    def get_user(self, user_id):
        return self.db_connection.query(f"SELECT * FROM users WHERE id = {user_id}")

class UserService:
    def __init__(self, user_repository):
        self.user_repository = user_repository
    
    def get_user_details(self, user_id):
        return self.user_repository.get_user(user_id)

# Dependency injection with a container
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    db_connection = providers.Singleton(
        Database,
        connection_string=config.db.connection_string,
    )
    
    user_repository = providers.Factory(
        DatabaseUserRepository,
        db_connection=db_connection,
    )
    
    user_service = providers.Factory(
        UserService,
        user_repository=user_repository,
    )

# Usage
container = Container()
container.config.db.connection_string.from_env("DATABASE_URL")
user_service = container.user_service()
user = user_service.get_user_details(1)
```

#### Domain-Driven Design

```python
# Value Object
@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str
    
    def __add__(self, other):
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)
    
    def __sub__(self, other):
        if self.currency != other.currency:
            raise ValueError("Cannot subtract different currencies")
        return Money(self.amount - other.amount, self.currency)

# Entity
class Order:
    def __init__(self, order_id: str, customer_id: str):
        self.order_id = order_id
        self.customer_id = customer_id
        self.items = []
        self.status = "new"
    
    def add_item(self, product_id: str, quantity: int, unit_price: Money):
        item = OrderItem(product_id, quantity, unit_price)
        self.items.append(item)
    
    def total_price(self) -> Money:
        if not self.items:
            return Money(Decimal('0'), 'USD')
        
        currency = self.items[0].unit_price.currency
        total = Decimal('0')
        
        for item in self.items:
            if item.unit_price.currency != currency:
                raise ValueError("Mixed currencies in order")
            total += item.unit_price.amount * item.quantity
        
        return Money(total, currency)
    
    def place(self):
        if not self.items:
            raise ValueError("Cannot place an empty order")
        self.status = "placed"

# Aggregate
class Cart:
    def __init__(self, cart_id: str, customer_id: str):
        self.cart_id = cart_id
        self.customer_id = customer_id
        self.items = {}  # product_id -> CartItem
    
    def add_item(self, product_id: str, quantity: int, unit_price: Money):
        if product_id in self.items:
            self.items[product_id].increase_quantity(quantity)
        else:
            self.items[product_id] = CartItem(product_id, quantity, unit_price)
    
    def remove_item(self, product_id: str):
        if product_id in self.items:
            del self.items[product_id]
    
    def checkout(self) -> Order:
        if not self.items:
            raise ValueError("Cannot checkout empty cart")
        
        order = Order(str(uuid.uuid4()), self.customer_id)
        for item in self.items.values():
            order.add_item(item.product_id, item.quantity, item.unit_price)
        
        order.place()
        self.items.clear()
        
        return order

# Repository
class OrderRepository:
    def save(self, order: Order):
        pass
    
    def find_by_id(self, order_id: str) -> Optional[Order]:
        pass
    
    def find_by_customer_id(self, customer_id: str) -> List[Order]:
        pass
```

### API Development

#### RESTful API with Flask

```python
from flask import Flask, request, jsonify
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class UserResource(Resource):
    def get(self, user_id):
        # Get user by ID
        user = get_user_from_db(user_id)
        if user:
            return jsonify(user)
        return {"error": "User not found"}, 404
    
    def put(self, user_id):
        # Update user
        data = request.get_json()
        success = update_user(user_id, data)
        if success:
            return {"message": "User updated successfully"}
        return {"error": "Failed to update user"}, 400
    
    def delete(self, user_id):
        # Delete user
        success = delete_user(user_id)
        if success:
            return {"message": "User deleted successfully"}
        return {"error": "Failed to delete user"}, 400

class UsersResource(Resource):
    def get(self):
        # List all users
        users = get_all_users()
        return jsonify(users)
    
    def post(self):
        # Create new user
        data = request.get_json()
        user_id = create_user(data)
        return {"message": "User created successfully", "user_id": user_id}, 201

# Register resources
api.add_resource(UsersResource, '/users')
api.add_resource(UserResource, '/users/<int:user_id>')

if __name__ == '__main__':
    app.run(debug=True)
```

#### FastAPI Example

```python
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session

from . import models, schemas
from .database import get_db

app = FastAPI()

class User(BaseModel):
    id: Optional[int] = None
    name: str
    email: str
    active: bool = True

@app.post("/users/", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = models.User(name=user.name, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(models.User).offset(skip).limit(limit).all()
    return users

@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.put("/users/{user_id}", response_model=schemas.User)
def update_user(user_id: int, user: schemas.UserUpdate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    for key, value in user.dict(exclude_unset=True).items():
        setattr(db_user, key, value)
    
    db.commit()
    db.refresh(db_user)
    return db_user

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(db_user)
    db.commit()
    return None
```

#### API Authentication

```python
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Change this in production!
jwt = JWTManager(app)

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    
    # Check username and password (implement proper authentication!)
    if username != 'test' or password != 'test':
        return jsonify({"message": "Invalid credentials"}), 401
    
    # Create access token
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    # Access the identity of the current user with get_jwt_identity
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200

if __name__ == '__main__':
    app.run()
```

### Database Interaction

#### SQLAlchemy ORM

```python
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

# Create engine and base
engine = create_engine('sqlite:///app.db', echo=True)
Base = declarative_base()

# Define models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    posts = relationship("Post", back_populates="author")
    
    def __repr__(self):
        return f"<User {self.username}>"

class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    author = relationship("User", back_populates="posts")
    
    def __repr__(self):
        return f"<Post {self.title}>"

# Create tables
Base.metadata.create_all(engine)

# Create session
Session = sessionmaker(bind=engine)