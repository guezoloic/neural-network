import math
import random
from typing import Optional, Callable


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def sigmoid_deriv(x: float) -> float:
    y: float = sigmoid(x)
    return y * (1 - y)


class Neuron:
    def __init__(self, input_size: int) -> None:
        """
        Set up the neuron's parameters (weights, and bias)

        :param input_size: Number of incomming inputs (must be > 0)
        """
        # Store input dimensions for structural consistency
        self._input_size: int = input_size

        # Scale each input influence using random weights.
        self._weight: list[float] = [
            random.uniform(-1., 1.) for _ in range(input_size)
        ]

        # Initialize a shift to the activation threshold with a random bias
        self._bias: float = random.uniform(-1., 1.)

    def forward(self, x: list[int], fn: Optional[Callable[[float], float]] = None) -> float:
        """
        Execute the neuron's forward pass.

        :param x: Description
        :param activate: Description
        """

        if len(x) != self._input_size:
            raise ValueError(
                f"Input vertex dimension {len(x)} mismatches the "
                f"stored size {self._input_size}")

        self._z: float = sum(welement * xelement for welement,
                             xelement in zip(self._weight, x)) + self._bias

        return fn(self._z) if fn is not None else self._z

    def backward(self, dz_dw: list[float], dcost_dy: float, learning_rate: float,
                 fn_deriv: Optional[Callable[[float], float]] = None) -> list[float]:
        # Check dimension consistency
        if len(dz_dw) != self._input_size:
            raise ValueError(
                f"Input vertex dimension {len(dz_dw)} mismatches "
                f"stored size {self._input_size}")

        # Local gradient: dy/dz (defaults to 1.0 for linear identity)
        dy_dz: float = fn_deriv(self._z) if fn_deriv is not None else 1.

        # Compute common error term (delta)
        # dC/dz = dC/dy * dy/dz
        delta: float = dcost_dy * dy_dz

        # Update weights: weight -= learning_rate * dC/dz * dz/dw
        for i in range(self._input_size):
            self._weight[i] -= learning_rate * delta * dz_dw[i]

        # Update bias: bias -= learning_rate * dC/dz * dz/db (where dz/db = 1)
        self._bias -= learning_rate * delta * 1.0

        # Return input gradient (dC/dx): dC/dz * dz/dx (where dz/dx = weights)
        # This vector allows the error to flow back to previous layers.
        return [delta * w for w in self._weight]

    def __repr__(self) -> str:
        jmp: int = int(math.sqrt(self._input_size))
        text: list[str] = []
        
        for i in range(0, self._input_size, jmp):
            line: str = str.join("", str(self._weight[i: (i + jmp)]))
            text.append(line)

        return "weight:\n" + str.join("\n", text) + f"\nbias: {self._bias}"

Neuron(10)

# # neuron class
# class Neuron:
#     """
#     z                   : linear combination of inputs and weights plus bias (pre-activation)
#     y                   : output of the activation function (sigmoid(z))
#     w                   : list of weights, one for each input
#     """
#     def __init__(self, input_size):
#         # number of inputs to this neuron
#         self._input_size = input_size
#         # importance to each input
#         self._weight = [random.uniform(-1, 1) for _ in range(self._input_size)]
#         # importance of the neuron
#         self._bias = random.uniform(-1, 1)

#     def forward(self, x, activate=True):
#         """
#         x               : list of input values to the neuron
#         """
#         # computes the weighted sum of inputs and add the bias
#         self._z = sum(w * xi for w, xi in zip(self.weight, x)) + self.bias
#         # normalize the output between 0 and 1 if activate
#         last_output = sigmoid(self.z) if activate else self.z

#         return last_output

#     # adjust weight and bias of neuron
#     def backward(self, x, dcost_dy, learning_rate):
#         """
#         x               : list of input values to the neuron
#         dcost_dy        : derivate of the cost function `(2 * (output - target))`
#         learning_rate   : learning factor (adjust the speed of weight/bias change during training)

#         weight -= learning_rate * dC/dy * dy/dz * dz/dw
#         bias   -= learning_rate * dC/dy * dy/dz * dz/db
#         """
#         # dy/dz: derivate of the sigmoid activation
#         dy_dz = sigmoid_deriv(self.z)
#         # dz/dw = x
#         dz_dw = x

#         assert len(dz_dw) >= self.isize, "too many value for input size"

#         # dz/db = 1
#         dz_db = 1

#         for i in range(self.isize):
#             # update each weight `weight -= learning_rate * dC/dy * dy/dz * x_i`
#             self.weight[i] -= learning_rate * dcost_dy * dy_dz * dz_dw[i]

#         # update bias: bias -= learning_rate * dC/dy * dy/dz * dz/db
#         self.bias -= learning_rate * dcost_dy * dy_dz * dz_db

#         # return gradient vector len(weight) dimension
#         return [dcost_dy * dy_dz * w for w in self.weight]

#     def __repr__(self):
#         pass

# class Layer:
#     def __init__(self, input_size, output_size):
#         """
#         input_size      : size of each neuron input
#         output_size     : size of neurons
#         """
#         self.size = output_size
#         # list of neurons
#         self.neurons = [Neuron(input_size) for _ in range(output_size)]

#     def forward(self, inputs, activate=True):
#         self.inputs = inputs
#         #  give the same inputs to each neuron in the layer
#         return [neuron.forward(inputs, activate) for neuron in self.neurons]

#     # adjust weight and bias of the layer (all neurons)
#     def backward(self, dcost_dy_list, learning_rate=0.1):
#         # init layer gradient vector len(input) dimention
#         input_gradients = [0.0] * len(self.inputs)

#         for i, neuron in enumerate(self.neurons):
#             dcost_dy = dcost_dy_list[i]
#             grad_to_input = neuron.backward(self.inputs, dcost_dy, learning_rate)

#             # accumulate the input gradients from all neurons
#             for j in range(len(grad_to_input)):
#                 input_gradients[j] += grad_to_input[j]

#         # return layer gradient
#         return input_gradients

# class NeuralNetwork:
#     def __init__(self, layer_size):
#         self.layers = [Layer(layer_size[i], layer_size[i+1]) for i in range(len(layer_size) - 1)]

#     def forward(self, inputs):
#         output = inputs
#         for i, layer in enumerate(self.layers):
#             activate = (i != len(self.layers) - 1)  # deactivate sigmoid latest neuron
#             output = layer.forward(output, activate=activate)
#         return output

#     def backward(self, inputs, targets, learning_rate=0.1):
#         """
#         target must be a list with the same length that the final layer
#         input
#         """
#         output = self.forward(inputs)

#         # computes the initial gradient of the cost function for each neuron
#         # by using Mean Squared Error's derivate: dC/dy = 2 * (output - target)
#         dcost_dy_list = [2 * (o - t) for o, t in zip(output, targets)]

#         grad = dcost_dy_list
#         for layer in reversed(self.layers):
#             # backpropagate the gradient of the layer to update weights and biases
#             grad = layer.backward(grad, learning_rate)

#         # return final gradient
#         return grad

# if __name__ == "__main__":
#     print("you might want to run main.py instead of network.py")
