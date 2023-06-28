# digitect

*what i cannot create, i do not understand* - richard feynman.

this quote is the inspiration for this project. i could just use tensorflow or another ml library, but i feel like making my own neural network would help me understand how they work. i should probably start learning linear algebra soon as well.

the plan is to build a image processing model that can correctly identify handwritten digits. the data will be sourced from kaggle's [MNIST training set](https://www.kaggle.com/competitions/digit-recognizer). the main python libraries i'm going to use are `numpy` and `pandas` for array operations.

by the way, this project is not my original work -- i closely followed samson zhang's [blogpost](https://www.samsonzhang.com/2020/11/24/understanding-the-math-behind-neural-networks-by-building-one-from-scratch-no-tf-keras-just-numpy) on making neural networks from scratch. i'm not attempting this project to create anything new, but to understand what has already been created before through a hands-on and '[learning by teaching](https://en.wikipedia.org/wiki/Learning_by_teaching)' approach.

table of contents:
* what is a neural network?
* drawing out the mnist neural network
	* input layer
	* hidden layer
	* output layer
* part 1: forward propagation
	* activation functions
		* rectified linear unit
		* softmax
	* weights and biases
* part 2: backward propagation
	* one-hot encoding
* part 3: adjusting weights and biases
* gradient descent
* results

## what is a neural network?

brain architecture is rather interesting -- neurons transmit data across synapses and connect to thousands of other neurons, forming connections. and when we learn, neurons start to create unique connections refered to as neuroplasticity. thank you ap psychology. 

now, if we wanted a computer to 'learn', why not give it its own neurons and have it create its own connections? thats what a neural network is: a brain-inspired way in which computers learn to process and interpret data. there's something beautiful about the application of natural phenomena to artificial intelligence.

## drawing out the MNIST neural network

in the mnist dataset, we are presented with 28x28 pixel images of digits, for a total of 784 pixels. from these 784 pixels, we need to arrive at one of 10 digits (0 to 9).

![training](/images/training.png)

since we have a starting and ending point, we can 'weave' a neural net between these two points. let's make a 2-layer nn.

![nn](/images/nn.png)

**input layer**

our input layer contains 784 nodes (neurons), with each node representing a pixel in the image. this is a 0th layer -- since we don't really do any analysis on this layer, we theoretically don't add it to our nn layer count. 

**hidden layer**

the first layer, our hidden layer, contains 10 nodes. the hidden layer is where the input data is transformed using activation functions, weights, and biases (i'll explain these soon). this layer allows the data to be broken down into very specific transformations with different functions and point us to a potential output.

**the output layer**

the second layer, our output later, also contains 10 nodes. this layer coalesces and points us to a concrete end result. in our neural network, each of the nodes in this layer are respective to a possible digit.

## part 1: forward propagation

during forward propagation, we traverse through the neural network from the input to the output layer. after we run through the input using our weights and biases, we can compute an initial output.

![fprop](/images/fprop.png)

we can model forward propagation the equations above ... but what is `ReLU()` and `softmax()`? let's talk about these activation functions.

### activation functions

activation functions help us get outputs from nodes based on their respective weights and biases. without activation functions, we would just have a complicated linear regression model. there are many types such as sigmoid, tanh, and linear activation functions -- we're going to be using a ReLU (rectified linear unit) and softmax activation function.

**rectified linear unit**

![relu](/images/relu.png)

this activation function is very simple and extremely popular. its basically `ReLU(x) = max(0, x)`, giving us a piecewise linear function which throws out any negative `x` value.

**softmax**

![softmax](/images/softmax.png)

the softmax activation function is more of a concluding activation function that we can apply on the output layer of a neural network with set amount of output values, so that we can calculate the probability that our input 'matches' a possible output.

### weights and biases

weights and biases are essentially the parameters of an activation function. weights determine how 'important' each input is in predicting an output value. we initialize the model with random weights and then adjust the weights as we train the model. for example, the model year of a car would probably bear a weight in the calculation of that car's price, but something like the hair color of the car's previous owner would probably bear little to no weight in that calculation.

bias is a constant that is added to the product of the input and its weight. the bias offsets the result such that the model doesn't purely rely on inputs and weights, which can result in the model overfitting the training data. by reducing the variance, the bias forces the mdoel to be more flexible and therefore have better generalization capabilities.

## part 2: backward propagation

during backward propagation, we go basically go the opposite direction that we went in part 1 -- from the output to the input layer. why? by doing so, we can check our model's prediction and its deviation from the correct result. from there, we can determine the degree to which we want to modify our weights and biases based on the following equations:

![bprop](/images/bprop.png)

### one-hot encoding

we use one-hot encoding to convert data such that it is easy for an algorithm to read it and create a prediction. one-hot refers to the process of shifting categorical values into new categorical columns and assigning either a binary value to serve as a 'checkbox' for a category. 

![onehot](/images/onehot.png)

one-hot encoding is important for the first equation in the backward propogation section. we need to properly subtract the correct result from our predictions so that we can adjust the weights and biases in the next step by our calculated errors.

## part 3: adjusting weights and biases

to adjust the weights and biases of the model, we can examine the following equations:

![adjust](/images/adjust.png)

after making these changes, we need to iterate through the model again with our new weights and biases to train it!

## gradient descent

the process of using backwards propagation to adjust our model is a form of gradient descent. the purpose of gradient descent is to, simply put, find the minimum value of a function. how? by slowly travelling 'down' the steepest gradients until you reach the local minimum. when we use backward propagation and make changes to the weights and biases, we are 'locating' the 'steepest' path to a minimum error value and selecting new weights and biases at that 'location'.

![gradient](/images/gradient.png)


## results

after running our MNIST neural network on training data, after about 500 iterations the model usually reports an accuracy of over 85% -- not bad! i'm sure there are several ways to increase the accuracy, such as finding a better starting point for the weights and biases instead of randomly assignign them, or even changing our activation function -- these are my future goals for when i revisit this project.

here is correct and incorrect prediction from our neural network:

![demo](/images/demo.png)
