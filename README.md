# handwritten-digits-recognizer
A simple neural network in python capable of recognizing handwritten digits

This project was made for academic purposes.
Feel free to use it.

Made by Murilo M. Grosso.

------------------------------------------------------------------------------------------------------

This netwotk uses the MNIST dataset to learn.
The database consists of 60,000 grayscale images of handwritten digits from 0 to 9, with 28x28 pixels each.

![080b85fa-6251-42d9-b069-a96ac276eefe](https://user-images.githubusercontent.com/102973750/222712150-c58040a2-09ea-4e40-ba55-79df1cc62687.png)

Link: http://yann.lecun.com/exdb/mnist/

------------------------------------------------------------------------------------------------------

mnist_loader.py: Convert the mnist compressed files to numpy arrays.

neural_network.py: The network. Its default format is: 
  784 neurons for the input layer (one neuron per pixel)
  1 hidden layer with 30 neurons
  10 neurons for the output layer (one neuron per digit)
To learn, the network uses the backpropagation algorithm to update the weights and biases.

train_network.py: Trains the network using the MNIST dataset.
> python3 train_network.py

app.py: Creates a drawing canvas to test the AI.
> python3 app.py
