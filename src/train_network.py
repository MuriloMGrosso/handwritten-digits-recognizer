import neural_network as neural
import data_manager

net = neural.Network()
#net = neural.Network([784,128,30,10], "DigitsRecognizer0.2")
net.load("DigitsRecognizer0.2")

training_data, test_data = data_manager.get_data()
net.train(training_data, 100, 100, 0.03, test_data)