# train.py

import numpy as np
import matplotlib.pyplot as plt
import pickle
from mnist_data import MnistData
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp

# Loading the MNIST Dataset
print("Loading the MNIST Dataset...")
mnist = MnistData()
(x_train, y_train), (x_test, y_test) = mnist.load()
print("Dataset Loaded.")
print("Training data shape:", x_train.shape)

# Creating the Network
network = TwoLayerNetWithBackProp(input_size=28*28, hidden_size=100, output_size=10)

# Setting the Training Parameters
iterations = 10000
train_size = x_train.shape[0]
batch_size = 16
lr = 0.01
iter_per_epoch = max(train_size // batch_size, 1)

train_losses = []
train_accs = []
test_accs = []

# Training the Model
print("Starting training...")
for i in range(iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    # Compute gradients
    grads = network.gradient(x_batch, y_batch)

    # Update parameters
    for key in ('w1', 'b1', 'w2', 'b2'):
        network.params[key] -= lr * grads[key]

    # Record loss for plotting
    train_losses.append(network.loss(x_batch, y_batch))

    # Evaluate accuracy at the end of each epoch
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        test_acc = network.accuracy(x_test, y_test)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"train acc, test acc: {train_acc}, {test_acc}")

# Plotting the Results
print("Training completed. Plotting results...")
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_accs))
plt.plot(x, train_accs, label='train acc', marker=markers['train'])
plt.plot(x, test_accs, label='test acc', linestyle='--', marker=markers['test'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# Saving Model Weights
my_weight_pkl_file = 'Model/manoj_weights.pkl'
with open(my_weight_pkl_file, 'wb') as f:
    print(f'Pickle: {my_weight_pkl_file} is being created.')
    pickle.dump(network.params, f)
    print('Done.')
