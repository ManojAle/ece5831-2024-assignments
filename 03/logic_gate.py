import numpy as np

class LogicGate:
    """
    Base class for all logic gates. This class is meant to be inherited by other gate classes.
    """

    def __init__(self, label):
        """
        Initializes the LogicGate with a label.
        
        Args:
        label (str): A string label to identify the logic gate.
        """
        self.label = label
        self.input1 = None
        self.input2 = None

    def set_inputs(self, input1, input2):
        """
        Sets the inputs for the logic gate. Accepts inputs as 0 or 1.
        
        Args:
        input1 (int): First input for the logic gate (0 or 1).
        input2 (int): Second input for the logic gate (0 or 1).
        """
        # Convert 0 or 1 to boolean scalar values
        self.input1 = np.array(input1, dtype=bool)
        self.input2 = np.array(input2, dtype=bool)

    def get_output(self):
        """
        Returns the output of the logic gate after computation.
        
        Returns:
        int: Output of the logic gate (0 or 1).
        """
        return self.perform_logic_gate()

    def perform_logic_gate(self):
        """
        Placeholder method to be implemented by subclasses to define the logic of the gate.
        """
        raise NotImplementedError('Subclasses should implement this')

    def _calculate_output(self, weights, bias):
        """
        Calculates the output of the gate based on input, weights, and bias.
        
        Args:
        weights (np.ndarray): Weights for the inputs.
        bias (float): Bias for the calculation.
        
        Returns:
        bool: True or False based on the logic calculation.
        """
        x = np.array([self.input1, self.input2])
        y = np.dot(x, weights) + bias
        return y > 0


class AndGate(LogicGate):
    """
    AND Gate class that inherits from LogicGate. Returns 1 if both inputs are 1, otherwise 0.
    """

    def perform_logic_gate(self):
        """
        Implements the logic for the AND gate using weights and a bias.
        
        Returns:
        int: Output of the AND gate (0 or 1).
        """
        weights = np.array([0.5, 0.5])
        bias = -0.7
        return int(self._calculate_output(weights, bias))


class OrGate(LogicGate):
    """
    OR Gate class that inherits from LogicGate. Returns 1 if at least one input is 1, otherwise 0.
    """

    def perform_logic_gate(self):
        """
        Implements the logic for the OR gate using weights and a bias.
        
        Returns:
        int: Output of the OR gate (0 or 1).
        """
        weights = np.array([1, 1])
        bias = -0.9
        return int(self._calculate_output(weights, bias))


class NandGate(LogicGate):
    """
    NAND Gate class that inherits from LogicGate. Returns 0 if both inputs are 1, otherwise 1.
    """

    def perform_logic_gate(self):
        """
        Implements the logic for the NAND gate using weights and a bias.
        
        Returns:
        int: Output of the NAND gate (0 or 1).
        """
        weights = np.array([-0.5, -0.5])
        bias = 0.7
        return int(self._calculate_output(weights, bias))


class NorGate(LogicGate):
    """
    NOR Gate class that inherits from LogicGate. Returns 1 if both inputs are 0, otherwise 0.
    """

    def perform_logic_gate(self):
        """
        Implements the logic for the NOR gate using weights and a bias.
        
        Returns:
        int: Output of the NOR gate (0 or 1).
        """
        weights = np.array([-1, -1])
        bias = 0.9
        return int(self._calculate_output(weights, bias))


class XorGate(LogicGate):
    """
    XOR Gate class that inherits from LogicGate. Returns 1 if one input is 1 and the other is 0, otherwise 0.
    """

    def perform_logic_gate(self):
        """
        Implements the logic for the XOR gate using a combination of OR, NAND, and AND gates.
        
        Returns:
        int: Output of the XOR gate (0 or 1).
        """
        # XOR is constructed using OR, NAND, and AND gates
        y1 = OrGate(self.label)
        y1.set_inputs(self.input1, self.input2)
        or_output = y1.get_output()

        y2 = NandGate(self.label)
        y2.set_inputs(self.input1, self.input2)
        nand_output = y2.get_output()

        final_and_gate = AndGate(self.label)
        final_and_gate.set_inputs(or_output, nand_output)
        return final_and_gate.get_output()

### Sample Usage for each gate
# if __name__ == "__main__":
#     # Test logic gates with scalar inputs 0 and 1
#     and_gate = AndGate("AND Gate")
#     and_gate.set_inputs(1, 1)
#     print(f"AND Gate Output: {and_gate.get_output()}")  
#     nand_gate = NandGate("NAND Gate")
#     nand_gate.set_inputs(1, 1)
#     print(f"NAND Gate Output: {nand_gate.get_output()}") 

#     or_gate = OrGate("OR Gate")
#     or_gate.set_inputs(0, 0)
#     print(f"OR Gate Output: {or_gate.get_output()}")  

#     nor_gate = NorGate("NOR Gate")
#     nor_gate.set_inputs(0, 0)
#     print(f"NOR Gate Output: {nor_gate.get_output()}")  

#     xor_gate = XorGate("XOR Gate")
#     xor_gate.set_inputs(1, 0)
#     print(f"XOR Gate Output: {xor_gate.get_output()}")  
