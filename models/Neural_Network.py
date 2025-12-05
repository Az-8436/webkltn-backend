import numpy as np
def ReLU(Z):
    return np.maximum(0, Z)
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
def forward_prop(X, W1, b1, W2, b2, W3, b3):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3