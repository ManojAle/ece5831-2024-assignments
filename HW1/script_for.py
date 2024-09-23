# script.py

# Importing the module bar from the package foo
import foo.bar

# Use the function defined in bar
greeting = foo.bar.hello()
print(greeting)
from foo import bar
print(bar.hello())