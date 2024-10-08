from multilayer_percetron import MultiLayerPerceptron as MLP
import numpy as np

if __name__ == '__main__':
    mlp = MLP()
    mlp.init_network()
    y = mlp.forward(np.array([3.3, 7.7]))
    print(y)
