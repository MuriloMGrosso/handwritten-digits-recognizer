import numpy as np
import random
from progress.bar import Bar
import data_manager
import time

class Network(object):
    def __init__(self, layers_sizes=[], name="NewNetwork"):        
        self.num_layers = len(layers_sizes)
        self.biases = [np.random.randn(y, 1) for y in layers_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers_sizes[:-1], layers_sizes[1:])]
        self.accuracy = 0
        self.name = name

    def feedforward(self,a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train(self, training_data, epochs, batch_size, learning_rate, test_data=None):
        maxAccuracy = self.accuracy
        for epoch in range(epochs):
            start_time = time.time()

            randomized_training_data = data_manager.randomize_data(training_data)
            random.shuffle(randomized_training_data)
            batches = [randomized_training_data[j:j + batch_size] for j in range(0, len(randomized_training_data), batch_size)]
            print("Epoch {0}/{1}:".format(epoch + 1, epochs))
            bar = Bar("Training", max=len(randomized_training_data))
            for batch in batches:
                self.update(batch, learning_rate)
                bar.next(batch_size)
            bar.finish()
            if test_data:
                self.accuracy = self.efficiency(data_manager.randomize_data(test_data))
                print("Accuracy: {0}%".format(round(self.accuracy * 100, 2)))

            end_time = time.time()
            elapsed_time = int(end_time - start_time)
            print("Took {0}s".format(elapsed_time))
            print("{0}s left".format(elapsed_time * (epochs - epoch - 1)))
            print("\n")

            if self.accuracy > maxAccuracy:
                maxAccuracy = self.accuracy
                self.save()

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

        fail_table = np.zeros(10)
        count_table = np.zeros(10)
        sucess = 0

        for (x,y) in test_results:
            count_table[y] += 1
            if int(x == y):
                sucess += 1
            else:
                fail_table[y] += 1

        print("Fail table: [", end=" ")
        for i in range(len(fail_table)):
            print("{0}% ".format(round(100 * fail_table[i] / count_table[i], 2)), end=" ")
        print("]")
        
        return sucess / len(test_data)

    def evaluate(self, data):
        output = self.feedforward(data)
        result = np.argmax(output)
        s = 0
        for n in output:
            s += n[0]
        print("Answer: {0} ({1}%)".format(result, round(output[result][0]/s * 100, 2)))
        return output / s
    
    def print(self):
        print(self.name)
        print("Accuracy: {0}%\n".format(round(self.accuracy * 100, 2)))

    def save(self):
        print("Saving...")
        np.save(f'../network_data/{self.name}.npy', np.array([self.weights, self.biases, [self.accuracy], [self.name]], dtype=object))
        print("Done!\n")

    def load(self, filename):
        print("Loading...")
        content = np.load(f'../network_data/{filename}.npy', allow_pickle=True)
        self.weights = content[0]
        self.biases = content[1]
        self.num_layers = len(self.biases) + 1
        self.accuracy = content[2][0]
        self.name = content[3][0]
        print("Done!\n")
        self.print()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))