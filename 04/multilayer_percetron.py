import numpy as np

class MultiLayerPerceptron:
    """
    A simple multi-layer perceptron (MLP) implementation with 3 layers.

    Attributes:
    -----------
    net : dict
        Dictionary containing weights ('w1', 'w2', 'w3') and biases ('b1', 'b2', 'b3') for each layer of the network.

    Methods:
    --------
    init_network():
        Initializes the network weights and biases for a 3-layer perceptron.
    
    forward(x):
        Performs a forward pass through the network using input `x`.
    
    identity(x):
        The identity activation function, used for the output layer.
    
    sigmoid(x):
        Sigmoid activation function, used for the hidden layers.
    """
    
    def __init__(self):
        """
        Initializes the MultiLayerPerceptron class. Creates an empty dictionary 'net' for storing network weights and biases.
        """
        self.net = {}
    
    def init_network(self):
        """
        Initializes the weights and biases of the network with predefined values for a 3-layer perceptron.
        
        Layer 1: Input -> Hidden layer 1
        Layer 2: Hidden layer 1 -> Hidden layer 2
        Layer 3: Hidden layer 2 -> Output layer
        """
        net = {}
        # layer 1
        net['w1'] = np.array([[0.7, 0.9, 0.3],[0.5, 0.4, 0.1]])
        net['b1'] = np.array([1, 1, 1])
        # layer 2
        net['w2'] = np.array([[0.2, 0.3], [0.4, 0.5], [0.22, 0.1234]])
        net['b2'] = np.array([0.5, 0.5])
        # layer 3 <-- output
        net['w3'] = np.array([[0.7, 0.1], [0.123, 0.314]])
        net['b3'] = np.array([0.1, 0.2])

        self.net = net
    
    def forward(self, x):
        """
        Performs the forward pass for the given input `x`.
        
        Parameters:
        -----------
        x : numpy.ndarray
            The input vector to the MLP.

        Returns:
        --------
        y : numpy.ndarray
            The output after passing the input through the MLP layers.
        """
        w1, w2, w3 = self.net['w1'], self.net['w2'], self.net['w3']
        b1, b2, b3 = self.net['b1'], self.net['b2'], self.net['b3']

        # Layer 1 forward pass
        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        # Layer 2 forward pass
        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        # Layer 3 (output layer) forward pass
        a3 = np.dot(z2, w3) + b3
        y = self.identity(a3)

        return y

    def identity(self, x):
        """
        Identity activation function.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input to the output layer.

        Returns:
        --------
        x : numpy.ndarray
            The output is the same as the input.
        """
        return x
    
    def sigmoid(self, x):
        """
        Sigmoid activation function.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input to a hidden layer.

        Returns:
        --------
        numpy.ndarray
            Output after applying the sigmoid function element-wise.
        """
        return 1 / (1 + np.exp(-x))

# To display help for this class:
#help(MultiLayerPerceptron)
