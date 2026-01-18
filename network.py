import math
import random

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def sigmoid_deriv(x: float) -> float:
    y: float = sigmoid(x)
    return y * (1 - y)

# neuron class
class Neuron:
    """
    z                   : linear combination of inputs and weights plus bias (pre-activation)
    y                   : output of the activation function (sigmoid(z))
    w                   : list of weights, one for each input
    """
    def __init__(self, isize):
        # number of inputs to this neuron
        self.isize = isize
        # importance to each input
        self.weight = [random.uniform(-1, 1) for _ in range(self.isize)]
        # importance of the neuron
        self.bias = random.uniform(-1, 1)

    def forward(self, x, activate=True):
        """
        x               : list of input values to the neuron
        """
        # computes the weighted sum of inputs and add the bias
        self.z = sum(w * xi for w, xi in zip(self.weight, x)) + self.bias
        # normalize the output between 0 and 1 if activate
        last_output = sigmoid(self.z) if activate else self.z

        return last_output
    
    # adjust weight and bias of neuron
    def backward(self, x, dcost_dy, learning_rate):
        """
        x               : list of input values to the neuron  
        dcost_dy        : derivate of the cost function `(2 * (output - target))`
        learning_rate   : learning factor (adjust the speed of weight/bias change during training)

        weight -= learning_rate * dC/dy * dy/dz * dz/dw
        bias   -= learning_rate * dC/dy * dy/dz * dz/db
        """
        # dy/dz: derivate of the sigmoid activation
        dy_dz = sigmoid_deriv(self.z)
        # dz/dw = x
        dz_dw = x

        assert len(dz_dw) >= self.isize, "too many value for input size"

        # dz/db = 1
        dz_db = 1

        for i in range(self.isize):
            # update each weight `weight -= learning_rate * dC/dy * dy/dz * x_i`
            self.weight[i] -= learning_rate * dcost_dy * dy_dz * dz_dw[i]

        # update bias: bias -= learning_rate * dC/dy * dy/dz * dz/db
        self.bias -= learning_rate * dcost_dy * dy_dz * dz_db

        # return gradient vector len(input) dimension
        return [dcost_dy * dy_dz * w for w in self.weight]


class Layer:
    def __init__(self, input_size, output_size):
        """
        input_size      : size of each neuron input
        output_size     : size of neurons
        """
        self.size = output_size
        # list of neurons
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward(self, inputs, activate=True):
        self.inputs = inputs
        #  give the same inputs to each neuron in the layer
        return [neuron.forward(inputs, activate) for neuron in self.neurons]

    # adjust weight and bias of the layer (all neurons)
    def backward(self, dcost_dy_list, learning_rate=0.1):
        # init layer gradient vector len(input) dimention
        input_gradients = [0.0] * len(self.inputs)

        for i, neuron in enumerate(self.neurons):
            dcost_dy = dcost_dy_list[i]
            grad_to_input = neuron.backward(self.inputs, dcost_dy, learning_rate)

            # accumulate the input gradients from all neurons
            for j in range(len(grad_to_input)):
                input_gradients[j] += grad_to_input[j]

        # return layer gradient
        return input_gradients

class NeuralNetwork:
    def __init__(self, layer_size):
        self.layers = [Layer(layer_size[i], layer_size[i+1]) for i in range(len(layer_size) - 1)]

    def forward(self, inputs):
        output = inputs
        for i, layer in enumerate(self.layers):
            activate = (i != len(self.layers) - 1)  # deactivate sigmoid latest neuron
            output = layer.forward(output, activate=activate)
        return output

    def backward(self, inputs, targets, learning_rate=0.1):
        """
        target must be a list with the same length that the final layer
        input
        """
        output = self.forward(inputs)

        # computes the initial gradient of the cost function for each neuron
        # by using Mean Squared Error's derivate: dC/dy = 2 * (output - target)
        dcost_dy_list = [2 * (o - t) for o, t in zip(output, targets)]

        grad = dcost_dy_list
        for layer in reversed(self.layers):
            # backpropagate the gradient of the layer to update weights and biases
            grad = layer.backward(grad, learning_rate)
        
        # return final gradient 
        return grad

if __name__ == "__main__":
    print("you might want to run main.py instead of network.py")