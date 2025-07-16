# Python Programming Basics: A Complete Guide

## Table of Contents
1. [Introduction to Python](#introduction)
2. [Variables and Data Types](#variables-and-data-types)
3. [Operators](#operators)
4. [Control Structures](#control-structures)
5. [Functions](#functions)
6. [Data Structures](#data-structures)
7. [Object-Oriented Programming](#object-oriented-programming)
8. [File Handling](#file-handling)
9. [Error Handling](#error-handling)
10. [Modules and Packages](#modules-and-packages)



Variables and Data Types
Operators
Control Flow (if, else, loops)
Functions
Lists, Tuples, and Dictionaries
Strings
Modules and Packages
File Handling
Exception Handling
Object-Oriented Programming (Classes and Objects)

## Introduction to Python {#introduction}

Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

### Why Python?
- **Easy to learn**: Simple, clean syntax
- **Versatile**: Web development, data science, AI, automation
- **Large community**: Extensive libraries and support
- **Cross-platform**: Runs on Windows, Mac, Linux

### Your First Python Program

```python
# This is a comment
print("Hello, World!")
```

Output:
```
Hello, World!
```

## Variables and Data Types {#variables-and-data-types}

Variables in Python are containers for storing data. You don't need to declare the type explicitly.

### Variable Assignment

```python
# Variable assignment
name = "Alice"
age = 25
height = 5.6
is_student = True

print(f"Name: {name}, Age: {age}, Height: {height}, Student: {is_student}")
```

### Basic Data Types

#### 1. Strings (str)
```python
# String examples
first_name = "John"
last_name = 'Doe'
full_name = first_name + " " + last_name

# String methods
message = "Hello, Python World!"
print(message.upper())        # HELLO, PYTHON WORLD!
print(message.lower())        # hello, python world!
print(message.split(","))     # ['Hello', ' Python World!']
print(len(message))           # 20
```

#### 2. Numbers (int, float)
```python
# Integer
count = 42
negative = -10

# Float
price = 19.99
temperature = -5.5

# Operations
result = count + price
print(f"Result: {result}")    # Result: 61.99
```

#### 3. Booleans (bool)
```python
is_active = True
is_complete = False

# Boolean operations
print(is_active and is_complete)  # False
print(is_active or is_complete)   # True
print(not is_active)              # False
```

#### 4. Type Checking and Conversion
```python
x = 42
print(type(x))           # <class 'int'>

# Type conversion
str_number = "123"
int_number = int(str_number)
float_number = float(str_number)

print(f"String: {str_number}, Int: {int_number}, Float: {float_number}")
```

## Operators {#operators}

### Arithmetic Operators
```python
a = 10
b = 3

print(f"Addition: {a + b}")        # 13
print(f"Subtraction: {a - b}")     # 7
print(f"Multiplication: {a * b}")  # 30
print(f"Division: {a / b}")        # 3.333...
print(f"Floor Division: {a // b}") # 3
print(f"Modulus: {a % b}")         # 1
print(f"Exponentiation: {a ** b}") # 1000
```

### Comparison Operators
```python
x = 5
y = 10

print(f"Equal: {x == y}")          # False
print(f"Not equal: {x != y}")      # True
print(f"Greater than: {x > y}")    # False
print(f"Less than: {x < y}")       # True
print(f"Greater or equal: {x >= y}") # False
print(f"Less or equal: {x <= y}")  # True
```

## Control Structures {#control-structures}

### If Statements
```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

print(f"Your grade is: {grade}")
```

### Loops

#### For Loops
```python
# Loop through a range
for i in range(5):
    print(f"Count: {i}")

# Loop through a list
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(f"I like {fruit}")

# Loop with enumerate (index and value)
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
```

#### While Loops
```python
count = 0
while count < 5:
    print(f"Count is {count}")
    count += 1

# While loop with user input
password = ""
while password != "secret":
    password = input("Enter password: ")
    if password != "secret":
        print("Wrong password!")
print("Access granted!")
```

#### Loop Control
```python
# Break and continue
for i in range(10):
    if i == 3:
        continue  # Skip this iteration
    if i == 7:
        break     # Exit the loop
    print(i)
```

## Functions {#functions}

Functions are reusable blocks of code that perform specific tasks.

### Basic Function Definition
```python
def greet(name):
    """This function greets someone"""
    return f"Hello, {name}!"

# Call the function
message = greet("Alice")
print(message)  # Hello, Alice!
```

### Function Parameters
```python
# Default parameters
def introduce(name, age=25, city="Unknown"):
    return f"Hi, I'm {name}, {age} years old from {city}"

print(introduce("Bob"))                    # Uses default age and city
print(introduce("Charlie", 30))            # Uses custom age, default city
print(introduce("Diana", 28, "New York"))  # All custom parameters
```

### Multiple Return Values
```python
def calculate(a, b):
    addition = a + b
    multiplication = a * b
    return addition, multiplication

sum_result, product_result = calculate(5, 3)
print(f"Sum: {sum_result}, Product: {product_result}")
```

### Lambda Functions (Anonymous Functions)
```python
# Lambda function
square = lambda x: x ** 2
print(square(5))  # 25

# Using lambda with built-in functions
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x ** 2, numbers))
print(squared_numbers)  # [1, 4, 9, 16, 25]
```

## Data Structures {#data-structures}

### Lists
Lists are ordered, mutable collections.

```python
# Creating lists
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = ["hello", 42, True, 3.14]

# List operations
fruits.append("grape")          # Add to end
fruits.insert(1, "kiwi")        # Insert at index
fruits.remove("banana")         # Remove by value
popped = fruits.pop()           # Remove and return last item

print(f"Fruits: {fruits}")
print(f"First fruit: {fruits[0]}")
print(f"Last fruit: {fruits[-1]}")
print(f"Slice: {fruits[1:3]}")
```

### Tuples
Tuples are ordered, immutable collections.

```python
# Creating tuples
coordinates = (10, 20)
person = ("Alice", 30, "Engineer")

# Tuple unpacking
x, y = coordinates
name, age, job = person

print(f"Coordinates: x={x}, y={y}")
print(f"Person: {name}, {age}, {job}")
```

### Dictionaries
Dictionaries store key-value pairs.

```python
# Creating dictionaries
student = {
    "name": "John",
    "age": 22,
    "grade": "A",
    "courses": ["Math", "Physics"]
}

# Dictionary operations
student["email"] = "john@email.com"    # Add new key-value
student["age"] = 23                    # Update existing value
del student["grade"]                   # Delete key-value pair

# Accessing values
print(f"Name: {student['name']}")
print(f"Age: {student.get('age', 'Unknown')}")

# Iterating through dictionary
for key, value in student.items():
    print(f"{key}: {value}")
```

### Sets
Sets are unordered collections of unique elements.

```python
# Creating sets
colors = {"red", "green", "blue"}
numbers = {1, 2, 3, 3, 4, 4, 5}  # Duplicates are automatically removed

print(f"Colors: {colors}")
print(f"Numbers: {numbers}")  # {1, 2, 3, 4, 5}

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print(f"Union: {set1 | set2}")         # {1, 2, 3, 4, 5, 6}
print(f"Intersection: {set1 & set2}")  # {3, 4}
print(f"Difference: {set1 - set2}")    # {1, 2}
```

# Python Data Structures Comparison

| Feature | **Lists** | **Tuples** | **Dictionaries** | **Sets** |
|---------|-----------|------------|------------------|----------|
| **Syntax** | `[]` | `()` | `{}` | `{}` or `set()` |
| **Mutability** | Mutable (can change) | Immutable (cannot change) | Mutable (can change) | Mutable (can change) |
| **Ordered** | Yes (maintains insertion order) | Yes (maintains insertion order) | Yes (Python 3.7+) | No (unordered) |
| **Duplicates** | Allowed | Allowed | Keys: No, Values: Yes | Not allowed |
| **Indexing** | Yes `list[0]` | Yes `tuple[0]` | By key `dict[key]` | No |
| **Data Storage** | Single values | Single values | Key-value pairs | Unique values only |

## Examples

### Lists
```python
# Creation
fruits = ["apple", "banana", "orange", "apple"]
numbers = [1, 2, 3, 4, 5]

# Access by index
print(fruits[0])        # apple
print(fruits[-1])       # apple (last item)

# Modification
fruits.append("grape")           # Add to end
fruits.insert(1, "kiwi")         # Insert at index 1
fruits.remove("banana")          # Remove first occurrence
fruits[0] = "cherry"             # Change by index

# Common operations
print(len(fruits))               # Length
print("apple" in fruits)         # Check membership
fruits.sort()                    # Sort in place
```

### Tuples
```python
# Creation
coordinates = (10, 20)
person = ("Alice", 25, "Engineer")
single_item = (42,)              # Note the comma for single item

# Access by index
print(coordinates[0])            # 10
print(person[-1])                # Engineer

# Unpacking
x, y = coordinates
name, age, job = person

# Cannot modify (immutable)
# coordinates[0] = 5             # This would cause an error!

# Common operations
print(len(person))               # Length
print("Alice" in person)         # Check membership
print(person.count("Alice"))     # Count occurrences
print(person.index(25))          # Find index of value
```

### Dictionaries
```python
# Creation
student = {
    "name": "John",
    "age": 22,
    "grade": "A",
    "courses": ["Math", "Physics"]
}

# Alternative creation
student2 = dict(name="Jane", age=20, grade="B")

# Access by key
print(student["name"])           # John
print(student.get("email", "N/A"))  # N/A (default value)

# Modification
student["email"] = "john@email.com"  # Add new key-value
student["age"] = 23                  # Update existing value
del student["grade"]                 # Delete key-value pair

# Common operations
print(student.keys())            # Get all keys
print(student.values())          # Get all values
print(student.items())           # Get key-value pairs

# Iteration
for key, value in student.items():
    print(f"{key}: {value}")
```

### Sets
```python
# Creation
colors = {"red", "green", "blue"}
numbers = {1, 2, 3, 4, 5}
empty_set = set()                # Note: {} creates an empty dict

# From list (removes duplicates)
unique_numbers = set([1, 2, 2, 3, 3, 4])  # {1, 2, 3, 4}

# Modification
colors.add("yellow")             # Add single item
colors.update(["purple", "orange"])  # Add multiple items
colors.remove("red")             # Remove item (error if not found)
colors.discard("pink")           # Remove item (no error if not found)

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print(set1 | set2)               # Union: {1, 2, 3, 4, 5, 6}
print(set1 & set2)               # Intersection: {3, 4}
print(set1 - set2)               # Difference: {1, 2}
print(set1 ^ set2)               # Symmetric difference: {1, 2, 5, 6}

# Common operations
print(len(colors))               # Length
print("blue" in colors)          # Check membership
```

## When to Use Each Data Structure

| Data Structure | **Best Used For** | **Example Use Cases** |
|----------------|-------------------|-----------------------|
| **Lists** | Ordered collections that need modification | Shopping lists, student grades, game scores |
| **Tuples** | Immutable sequences, fixed data | Coordinates (x, y), RGB colors (255, 128, 0), database records |
| **Dictionaries** | Key-value relationships, fast lookups | User profiles, configuration settings, word counts |
| **Sets** | Unique collections, mathematical operations | Removing duplicates, membership testing, set mathematics |

## Performance Characteristics

| Operation | **Lists** | **Tuples** | **Dictionaries** | **Sets** |
|-----------|-----------|------------|------------------|----------|
| **Access by index/key** | O(1) | O(1) | O(1) average | N/A |
| **Search (membership test)** | O(n) | O(n) | O(1) average | O(1) average |
| **Insert at end** | O(1) amortized | N/A | O(1) average | O(1) average |
| **Insert at beginning** | O(n) | N/A | N/A | N/A |
| **Delete** | O(n) | N/A | O(1) average | O(1) average |

## Common Methods Summary

### Lists
- `append()`, `insert()`, `remove()`, `pop()`, `sort()`, `reverse()`, `extend()`, `clear()`

### Tuples
- `count()`, `index()` (Limited methods due to immutability)

### Dictionaries
- `get()`, `keys()`, `values()`, `items()`, `pop()`, `update()`, `clear()`, `setdefault()`

### Sets
- `add()`, `remove()`, `discard()`, `pop()`, `clear()`, `union()`, `intersection()`, `difference()`

## Object-Oriented Programming {#object-oriented-programming}

### Classes and Objects
```python
class Car:
    # Class variable
    wheels = 4
    
    def __init__(self, make, model, year):
        # Instance variables
        self.make = make
        self.model = model
        self.year = year
        self.is_running = False
    
    def start(self):
        self.is_running = True
        return f"{self.make} {self.model} is now running!"
    
    def stop(self):
        self.is_running = False
        return f"{self.make} {self.model} has stopped."
    
    def __str__(self):
        return f"{self.year} {self.make} {self.model}"

# Creating objects
my_car = Car("Toyota", "Camry", 2020)
print(my_car)                    # 2020 Toyota Camry
print(my_car.start())            # Toyota Camry is now running!
```

### Inheritance
```python
class ElectricCar(Car):
    def __init__(self, make, model, year, battery_capacity):
        super().__init__(make, model, year)
        self.battery_capacity = battery_capacity
        self.charge_level = 100
    
    def charge(self):
        self.charge_level = 100
        return f"{self.make} {self.model} is fully charged!"
    
    def get_range(self):
        return self.battery_capacity * 3  # Simplified calculation

# Using inheritance
tesla = ElectricCar("Tesla", "Model 3", 2022, 75)
print(tesla.start())         # Inherited method
print(tesla.charge())        # New method
print(f"Range: {tesla.get_range()} miles")
```

## File Handling {#file-handling}

### Reading and Writing Files
```python
# Writing to a file
with open("example.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is a test file.\n")

# Reading from a file
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

# Reading line by line
with open("example.txt", "r") as file:
    for line_number, line in enumerate(file, 1):
        print(f"Line {line_number}: {line.strip()}")
```

### Working with CSV Files
```python
import csv

# Writing CSV
data = [
    ["Name", "Age", "City"],
    ["Alice", "25", "New York"],
    ["Bob", "30", "San Francisco"]
]

with open("people.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

# Reading CSV
with open("people.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
```

## Error Handling {#error-handling}

### Try-Except Blocks
```python
def divide_numbers(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return "Error: Cannot divide by zero!"
    except TypeError:
        return "Error: Invalid input type!"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    finally:
        print("Division operation completed.")

# Testing error handling
print(divide_numbers(10, 2))    # 5.0
print(divide_numbers(10, 0))    # Error: Cannot divide by zero!
print(divide_numbers("10", 2))  # Error: Invalid input type!
```

### Custom Exceptions
```python
class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def validate_age(age):
    if age < 0:
        raise CustomError("Age cannot be negative!")
    if age > 150:
        raise CustomError("Age seems unrealistic!")
    return f"Age {age} is valid."

try:
    print(validate_age(25))    # Age 25 is valid.
    print(validate_age(-5))    # Raises CustomError
except CustomError as e:
    print(f"Validation Error: {e.message}")
```

## Modules and Packages {#modules-and-packages}

### Using Built-in Modules
```python
import math
import random
import datetime

# Math module
print(f"Pi: {math.pi}")
print(f"Square root of 16: {math.sqrt(16)}")

# Random module
print(f"Random number: {random.randint(1, 10)}")
print(f"Random choice: {random.choice(['apple', 'banana', 'orange'])}")

# Datetime module
now = datetime.datetime.now()
print(f"Current time: {now}")
print(f"Formatted date: {now.strftime('%Y-%m-%d %H:%M:%S')}")
```

### Creating Your Own Module
```python
# Save this as my_utils.py
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

PI = 3.14159

# Using your module (in another file)
# import my_utils
# print(my_utils.greet("Alice"))
# print(my_utils.add(5, 3))
```

### Different Import Methods
```python
# Import entire module
import math

# Import specific functions
from math import sqrt, pi

# Import with alias
import datetime as dt

# Import all (not recommended)
from math import *

# Using the imports
print(sqrt(25))           # 5.0
print(pi)                 # 3.141592653589793
print(dt.datetime.now())  # Current datetime
```

## Summary and Best Practices

### Key Python Concepts Covered
1. **Variables and Data Types**: Understanding strings, numbers, booleans
2. **Control Flow**: if/elif/else, for/while loops
3. **Functions**: Defining reusable code blocks
4. **Data Structures**: Lists, tuples, dictionaries, sets
5. **OOP**: Classes, objects, inheritance
6. **File Handling**: Reading and writing files
7. **Error Handling**: Try/except blocks
8. **Modules**: Using and creating reusable code

### Python Best Practices
- **Use meaningful variable names**: `user_age` instead of `ua`
- **Follow PEP 8**: Python's style guide for consistent code
- **Use comments and docstrings**: Document your code
- **Keep functions small**: One function should do one thing
- **Handle errors appropriately**: Use try/except blocks
- **Use virtual environments**: Keep project dependencies separate

### Next Steps
1. Practice with small projects (calculator, to-do list, etc.)
2. Learn about popular libraries (requests, pandas, matplotlib)
3. Explore web frameworks (Flask, Django)
4. Study data science tools (NumPy, pandas, scikit-learn)
5. Build real-world applications

Remember: The best way to learn Python is by practicing. 
Start with small programs and gradually build more complex projects!
