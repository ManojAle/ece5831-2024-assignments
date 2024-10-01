# Logic Gates Project

This project implements and tests various logic gates (AND, NAND, OR, NOR, XOR) using Python and NumPy.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [How to Run](#how-to-run)
   - [Running logic_gate.py](#running-logic_gatepy)
   - [Running module3.py](#running-module3py)
   - [Using module3.ipynb](#using-module3ipynb)
4. [Requirements](#requirements)
5. [GitHub Repository Structure](#github-repository-structure)

## Overview

This project simulates logic gates using a Python class `LogicGate`. The logic gates are implemented using NumPy to handle the input logic. The following gates are implemented:
- AND Gate
- NAND Gate
- OR Gate
- NOR Gate
- XOR Gate

Each gate is tested to ensure that it returns the correct outputs based on its truth table.

## Project Structure

```
ece5831-yyyy-assignments/
└── 03/
    ├── __pycache__/              # Compiled Python files (automatically generated)
    ├── logic_gate.py             # Logic gate implementations
    ├── module3.py                # Test script for logic gates
    ├── module3.ipynb             # Jupyter notebook for documentation and usage
```

### Files:
- **`logic_gate.py`**: Contains the implementation of the `LogicGate` class and the logic gate subclasses (AND, NAND, OR, NOR, XOR).
- **`module3.py`**: A script that tests the gates implemented in `logic_gate.py`.
- **`module3.ipynb`**: A Jupyter notebook that includes code explanations, usage examples, and documentation of the logic gate class and its functions.

## How to Run

### Running `logic_gate.py`

This script contains the implementation of the logic gates. When executed, it shows help messages on how to use the `LogicGate` class and its subclasses.

To run the script:

```bash
$ python logic_gate.py
```

This will display a help message explaining how the logic gates work.

### Running `module3.py`

This script tests the `LogicGate` class to verify that the gates (AND, NAND, OR, NOR, XOR) are working correctly.

To run the tests:

```bash
$ python module3.py
```

This will output the results of the test cases for each gate.

### Using `module3.ipynb`

The `module3.ipynb` notebook provides documentation, code explanations, and examples of how to use the logic gates.

To open the notebook:

```bash
$ jupyter notebook module3.ipynb
```

This will open the notebook in your web browser, allowing you to interactively view and run the code.

## Requirements

Make sure the following packages are installed:

- Python 3.x
- NumPy
- Pandas (for testing in `module3.py`)

You can install these dependencies using `pip`:

```bash
pip install numpy pandas
```

Include the following files in the `03` directory:
- `logic_gate.py`
- `module3.py`
- `module3.ipynb`

