# handwritten-digits-recognizer
A simple neural network in python capable of recognizing handwritten digits

------------------------------------------------------------------------------------------------------

This netwotk uses the MNIST dataset to learn.
The database consists of 60,000 grayscale images of handwritten digits from 0 to 9, 28x28 pixels each.

![080b85fa-6251-42d9-b069-a96ac276eefe](https://user-images.githubusercontent.com/102973750/222712150-c58040a2-09ea-4e40-ba55-79df1cc62687.png)

Link: http://yann.lecun.com/exdb/mnist/

------------------------------------------------------------------------------------------------------

mnist_loader.py: Convert the mnist compressed files to numpy arrays and randomize the data.

neural_network.py: The network. Its default format is: 
- 784 neurons for the input layer (one neuron per pixel)
- 2 hidden layers: one with 128 neurons and other with 30
- 10 neurons for the output layer (one neuron per digit)

To learn, the it uses the backpropagation algorithm to update the weights and biases.

train_network.py: Trains the network using the MNIST dataset.
> python3 train_network.py

app.py: Creates a canvas to test the AI. The network will try to predict the user's drawing in real time.
> python3 app.py

There's a folder named "network_data" where the network data is saved.

------------------------------------------------------------------------------------------------------

REFERENCES:
- http://neuralnetworksanddeeplearning.com/chap1.html
- http://yann.lecun.com/exdb/mnist/

------------------------------------------------------------------------------------------------------

This project was made for academic purposes.
Feel free to use it.

Made by Murilo M. Grosso.
