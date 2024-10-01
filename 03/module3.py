import pandas as pd
from logic_gate import AndGate, NandGate, OrGate, NorGate, XorGate

test_cases = [
    # AND Gate
    {"gate": "AND", "inputs": (1, 1), "expected": 1},
    {"gate": "AND", "inputs": (1, 0), "expected": 0},
    {"gate": "AND", "inputs": (0, 1), "expected": 0},
    {"gate": "AND", "inputs": (0, 0), "expected": 0},

    # NAND Gate
    {"gate": "NAND", "inputs": (1, 1), "expected": 0},
    {"gate": "NAND", "inputs": (1, 0), "expected": 1},
    {"gate": "NAND", "inputs": (0, 1), "expected": 1},
    {"gate": "NAND", "inputs": (0, 0), "expected": 1},

    # OR Gate
    {"gate": "OR", "inputs": (1, 1), "expected": 1},
    {"gate": "OR", "inputs": (1, 0), "expected": 1},
    {"gate": "OR", "inputs": (0, 1), "expected": 1},
    {"gate": "OR", "inputs": (0, 0), "expected": 0},

    # NOR Gate
    {"gate": "NOR", "inputs": (1, 1), "expected": 0},
    {"gate": "NOR", "inputs": (1, 0), "expected": 0},
    {"gate": "NOR", "inputs": (0, 1), "expected": 0},
    {"gate": "NOR", "inputs": (0, 0), "expected": 1},

    # XOR Gate
    {"gate": "XOR", "inputs": (1, 1), "expected": 0},
    {"gate": "XOR", "inputs": (1, 0), "expected": 1},
    {"gate": "XOR", "inputs": (0, 1), "expected": 1},
    {"gate": "XOR", "inputs": (0, 0), "expected": 0},
]

# Function to get the actual output from the gate
def get_output(gate, inputs):
    if gate == "AND":
        gate_obj = AndGate("AND")
    elif gate == "NAND":
        gate_obj = NandGate("NAND")
    elif gate == "OR":
        gate_obj = OrGate("OR")
    elif gate == "NOR":
        gate_obj = NorGate("NOR")
    elif gate == "XOR":
        gate_obj = XorGate("XOR")
    else:
        raise ValueError("Unknown gate type")
    
    gate_obj.set_inputs(inputs[0], inputs[1])
    return gate_obj.get_output()

# Check each test case
results = []
for test in test_cases:
    actual = get_output(test["gate"], test["inputs"])
    results.append({
        "gate": test["gate"],
        "inputs": test["inputs"],
        "expected": test["expected"],
        "actual": actual,
        "pass": actual == test["expected"]
    })

# Display the results in a DataFrame
df = pd.DataFrame(results)
print(df)
