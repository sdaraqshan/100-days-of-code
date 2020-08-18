KERAS:

-A high levelneural network library written in Python running on Tensorflow.

-Use Keras when you need a Deep Learning library:
1. To run seamlessly on CPU and GPU.
2. Supports Convolution Networks and Recurrent Networks.
3. Easy and fast prototyping.

- Core data structure in Keras is a MODEL.
It is a way to organize layers.

-The main layer is Sequential which is a linear stack of layers.

-Layers are basically building blocks of neural networks in Keras.

- Contains tensor-in-tensor-out computation function.

-Flatten layer flattens the input.
It removes all of the Dimensions except for one.
Activation     (None,3,16)
Flatten        (None,48)

-Dense layer is a layer of neurons.
Each neuron receives input from all previous layer neurons,thus densely connected.
It has Weight Matrix W, Bias Vector b, Activation of Previous Layer a.

-Optimizers are algorithms or methods used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses.
Optimization algorithms or strategies are responsible for reducing the losses and to provide the most accurate results possible.
Adam is an adaptive learning rate optimization algorithm that's been designed specifically for training deep neural networks.
The algorithms leverages the power of adaptive learning rates methods to find individual learning rates for each parameter.

-Cross entropy is a loss function, used to measure the dissimilarity between the distribution of observed class labels and the predicted probabilities of class membership.
Categorical refers to the possibility of having more than two classes (instead of binary, which refers to two classes). 
Sparse refers to using a single integer from zero to the number of classes minus one (e.g. { 0; 1; or 2 } for a class label for a three-class problem)

-A metric is a function that is used to judge the performance of your model.

-The activation function is a mathematical “gate” in between the input feeding the current neuron and its output going to the next layer. 
It can be as simple as a step function that turns the neuron output on and off, depending on a rule or threshold.

-ReLU stands for rectified linear unit, and is a type of activation function. Mathematically, it is defined as y = max(0, x).

-Non-trainable parameters of a model are those that you will not be updating and optimized during training, and that have to be defined a priori, or passed as inputs.
The example of such parameters are: the number of hidden layers. nodes on each hidden layer.

- EPochs are "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation. 
When using validation_data or validation_split with the fit method of Keras models, evaluation will be run at the end of every epoch.

-By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
verbose=0 will show you nothing (silent)
verbose=1 will show you an animated progress bar like this: [=================================]
verbose=2 will just mention the number of epoch like this: Epoch[1/10]

-Softmax is very useful because it converts the scores to a normalized probability distribution, which can be displayed to a user or used as input to other systems.
For this reason it is usual to append a softmax function as the final layer of the neural network.

