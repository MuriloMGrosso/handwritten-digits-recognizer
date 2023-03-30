import neural_network as neural
import data_manager

net = neural.Network()
#net = neural.Network([784,128,32,10], "DigitsRecognizer")
net.load("DigitsRecognizer")

training_data, test_data = data_manager.get_data()
net.train(training_data, 480, 10, 3.0, test_data)