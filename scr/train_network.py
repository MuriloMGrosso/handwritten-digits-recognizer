import neural_network as neural
import mnist_loader as mnist

net = neural.Network()
net.load('../network_data/DigitsRecognizer.npy')

training_data, test_data = mnist.get_data()
net.train(training_data, 30, 10, 3.0, test_data)
net.save()