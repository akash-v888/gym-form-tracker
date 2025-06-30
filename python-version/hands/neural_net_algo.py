import numpy as np
import pickle

class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        # Initialize weights and biases
        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return z > 0

    def one_hot(self, y, num_classes):
        one_hot_labels = np.zeros((len(y), num_classes))
        for idx, label in enumerate(y):
            one_hot_labels[idx][label] = 1
        return one_hot_labels

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true):
        m = X.shape[0]
        y_onehot = self.one_hot(y_true, self.b2.shape[1])

        dz2 = self.a2 - y_onehot
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if (epoch+1) % 100 == 0:
                acc = self.evaluate(X, y)
                print(f"Epoch {epoch+1}: Accuracy = {acc:.2f}")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)
