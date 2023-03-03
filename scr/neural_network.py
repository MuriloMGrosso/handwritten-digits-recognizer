import numpy as np
import random
from progress.bar import Bar

class Network(object):
    def __init__(self, layers_sizes=[]):
        self.num_layers = len(layers_sizes)
        self.biases = [np.random.randn(y, 1) for y in layers_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers_sizes[:-1], layers_sizes[1:])]
        self.accuracy = 0

    def feedforward(self,a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train(self, training_data, epochs, batch_size, learning_rate, test_data=None):
        for epoch in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[j:j + batch_size] for j in range(0, len(training_data), batch_size)]
            print('Epoch {0}/{1}:'.format(epoch + 1, epochs))
            bar = Bar('Training', max=len(training_data))
            for batch in batches:
                self.update(batch, learning_rate)
                bar.next(batch_size)
            bar.finish()
            if test_data:
                self.accuracy = self.efficiency(test_data)
                print('Accuracy: {0}%\n'.format(round(self.accuracy * 100, 2)))

    def update(self, batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate/len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate/len(batch)) * nb for b, nb in zip(self.biases, nabla_b)] 

    def backprop(self, input, expected_output):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = input
        activations = [np.array(activation)]
        for b, w in zip(self.biases, self.weights):
            activation = sigmoid(np.dot(w, activation) + b)
            activations.append(activation)
        #backward
        delta = (activations[-1] - expected_output) * activations[-1] * (1 - activations[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            delta = np.dot(self.weights[-i+1].transpose(), delta) * activations[-i] * (1 - activations[-i])
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return(nabla_b, nabla_w)

    def efficiency(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) / len(test_data)

    def evaluate(self, data):
        output = self.feedforward(data)
        result = np.argmax(output)
        s = 0
        for n in output:
            s += n[0]
        print("Answer: {0} ({1}%)".format(result, round(output[result][0]/s * 100, 2)))
    
    def print(self):
        print("Accuracy: ", self.accuracy)
        print("Weights: ", self.weights)
        print("Biases: ", self.biases)

    def save(self):
        print("Saving...")
        np.save('../network_data/DigitsRecognizer.npy', np.array([self.weights, self.biases, [self.accuracy]], dtype=object))
        print("Done!\n")

    def load(self, filename):
        print("Loading...")
        content = np.load(filename, allow_pickle=True)
        self.weights = content[0]
        self.biases = content[1]
        self.num_layers = len(self.biases) + 1
        self.accuracy = content[2][0]
        print("Done!\n")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))