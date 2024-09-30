import numpy as np

class LogicGate:
    def __init__(self, label):
        self.label = label
        self.input1 = None
        self.input2 = None

    def set_inputs(self, input1, input2):
        # Convert 0 or 1 to boolean
        self.input1 = np.array(input1, dtype=bool)
        self.input2 = np.array(input2, dtype=bool)

    def get_output(self):
        return self.perform_logic_gate()

    def perform_logic_gate(self):
        raise NotImplementedError('Subclasses should implement this')


class AndGate(LogicGate):
    def perform_logic_gate(self):
        weights = np.array([0.5, 0.5])
        bias = -0.7
        return int(self._calculate_output(weights, bias))

    def _calculate_output(self, weights, bias):
        x = np.array([self.input1, self.input2])
        y = np.dot(x, weights) + bias
        return y > 0


class OrGate(LogicGate):
    def perform_logic_gate(self):
        weights = np.array([1, 1])
        bias = -0.9
        return int(self._calculate_output(weights, bias))

    def _calculate_output(self, weights, bias):
        x = np.array([self.input1, self.input2])
        y = np.dot(x, weights) + bias
        return y > 0


class NandGate(LogicGate):
    def perform_logic_gate(self):
        weights = np.array([-0.5, -0.5])
        bias = 0.7
        return int(self._calculate_output(weights, bias))

    def _calculate_output(self, weights, bias):
        x = np.array([self.input1, self.input2])
        y = np.dot(x, weights) + bias
        return y > 0


class NorGate(LogicGate):
    def perform_logic_gate(self):
        weights = np.array([-1, -1])
        bias = 0.9
        return int(self._calculate_output(weights, bias))

    def _calculate_output(self, weights, bias):
        x = np.array([self.input1, self.input2])
        y = np.dot(x, weights) + bias
        return y > 0


class XorGate(LogicGate):
    def perform_logic_gate(self):
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


if __name__ == "__main__":
    # Test logic gates with 0 and 1 as inputs
    and_gate = AndGate("AND Gate")
    and_gate.set_inputs([1, 0], [1, 1])
    print(f"AND Gate Output: {and_gate.get_output()}")  # Expected output: 0

    nand_gate = NandGate("NAND Gate")
    nand_gate.set_inputs([1, 0], [1, 1])
    print(f"NAND Gate Output: {nand_gate.get_output()}")  # Expected output: 1

    or_gate = OrGate("OR Gate")
    or_gate.set_inputs([1, 0], [0, 0])
    print(f"OR Gate Output: {or_gate.get_output()}")  # Expected output: 1

    nor_gate = NorGate("NOR Gate")
    nor_gate.set_inputs([0, 0], [0, 0])
    print(f"NOR Gate Output: {nor_gate.get_output()}")  # Expected output: 1

    xor_gate = XorGate("XOR Gate")
    xor_gate.set_inputs([1, 0], [0, 1])
    print(f"XOR Gate Output: {xor_gate.get_output()}")  # Expected output: 1
