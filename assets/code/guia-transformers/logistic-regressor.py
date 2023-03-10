#%%

# Jupyter code cell: https://code.visualstudio.com/docs/python/jupyter-support-py

# Modifications by @jaspock

from sklearn.datasets import make_blobs  # pip install scikit-learn
from matplotlib import pyplot as plt
import numpy as np
import torch

# logistic regression for binary classification
# adapted from https://github.com/ConsciousML/Logistic-Regression-from-scratch

class_centers = [ [2,3], [4,3] ]
samples = 100
dev = 2.7
training_steps = 5000
training_log_steps = training_steps / 10
learning_rate = 0.01
FEATURES = 2

# generate 2-dimensional classification dataset (2 classes):
X, y = make_blobs(n_samples=samples, centers=class_centers, cluster_std=dev, 
        n_features=FEATURES, random_state=4)
# X: [train_samples, 2], y: [train_samples], y contains 0s and 1s

# check if cuda or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (f'Using {device}')

# move data to tensors:
X = torch.from_numpy(np.float32(X)).to(device) # default numpy is float32
y = torch.from_numpy(np.float32(y)).to(device)

# create train/test views of the data:
mask = torch.ones(X.shape[0], dtype=bool).to(device)
mask[::3] = 0  # one every n elements for testing
X_train, y_train = X[mask], y[mask]
X_test, y_test = X[torch.logical_not(mask)], y[torch.logical_not(mask)]  # also ~mask or bitwise_not(mask)

# print(X_train, y_train)
# print(X_test, y_test)

def sigmoid(x):
    # x: torch.Size([train_samples])
    return 1 / (1 + torch.exp(-x))

def forward(X, weights, bias):
    # X: [train_samples, 2], weights: [2], bias: [1]
    # torch.matmul(X,weights): [train_samples], returns: [train_samples]

    """ torch.mm only works with compatible 2-dimensional tensors (matrices) and does not broadcast;
    torch.matmul(a,b) performs matrix-vector product if a is 2-dimensional and b is 1-dimensional;
    torch.matmul(a,b) prepends a 1 to the dimensions of a for the purpose of the matrix multiply if 
    a is 1-dimensional and b is 2-dimensional; """

    return sigmoid(torch.matmul(X,weights) + bias)  # bias is broadcasted to [train_samples]

def binary_cross_entropy(y_truth, y_pred):
    # y_truth: [train_samples], y_pred: [train_samples], m: float, returns: float (scalar)
    m = 1 / y_truth.shape[0]
    return -m * (y_truth * torch.log(y_pred) + 
                    (1 - y_truth) * torch.log(1 - y_pred)).sum()

def train(X, y_truth, weights, bias, lr=0.01, it=1000, it_log=100):
    # X: [train_samples, 2], y_truth: [train_samples], weights: [2], bias: [1]
    # err: [train_samples], grad_w: [2], grad_b: [1], bn_train: float
    # bias could be a scalar tensor or a scalar, instead of a 1-dimensional tensor with 1 element
    # tensors will be more convenient later when we use PyTorch to compute gradients
    for i in range(it):
        y_pred = forward(X, weights, bias)
        err = (y_pred - y_truth)
        # torch.mul (operator *) performs element-wise multiplication with broadcasting
        grad_w = (1 / y_truth.shape[0]) * torch.matmul(err, X)
        grad_b = (1 / y_truth.shape[0]) * torch.sum(err)
        weights = weights - lr * grad_w
        bias = bias - lr * grad_b
        if (i) % it_log == 0:
            bn_train = binary_cross_entropy(y_truth, y_pred).item()
            print (f'Epoch [{i+1}/{it}], Loss: {bn_train:.2f}')
    bn_train = binary_cross_entropy(y_truth, y_pred).item()
    return weights, bias, bn_train

# training the model
# if operands are on GPU, result of any operation is on GPU
# therefore, we just need to move weights and bias to GPU 
weights = torch.rand(X_train.shape[1], dtype=torch.float).to(device)
bias = torch.rand(1, dtype=torch.float).to(device)  # bias is a scalar tensor
weights, bias, bn_train = train(X_train, y_train, weights, bias, 
                            lr=learning_rate, it=training_steps, it_log=training_log_steps)
y_pred = forward(X_test, weights, bias)
print('Weights:', weights, 'Bias:', bias)
print(f'Binary CE on the train set: {bn_train:.2f}')

# decision boundary hyperplane equation:
# sigma(w1 * x1 + w2 * x2 + b) = 0.5
# 1 = exp(-(w1 * x1 + w2 * x2 + b))
# log 1 = log exp(-(w1 * x1 + w2 * x2 + b))
# 0 = -(w1 * x1 + w2 * x2 + b)
# x2 = -w1/w2 * x1 - b/w2 
# x2 = r*x1 + t
r = -weights[0].item() / weights[1].item()
t = (-bias.item()) / weights[1].item()

# compute accuracy with PyTorch:
prediction = forward(X_test,weights,bias) > 0.5
correct = prediction == y_test
accuracy = (torch.sum(correct) / y_test.shape[0])*100
print (f'Accuracy: {accuracy:.2f}%')

# plot the data and the line (hyperplane) for the decision boundary:
X = X.cpu()
y = y.cpu()
plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), s=50, c = y)  # s is size of the points, c is an array of colors
plt.title(f"Hyperplane learned by the logistic regressor")
plt.xlabel("x")
plt.ylabel("y")
x_hyperplane = np.linspace(0,6,100)
y_hyperplane = r*x_hyperplane+t
plt.plot(x_hyperplane, y_hyperplane, '-r', label='y=2x+1')  # -r means solid red line
plt.show()
#plt.savefig("decision-boundary.png")
