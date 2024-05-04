import numpy as np


def compute_error_lineal(x, y, w):
    return

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_act(x):
    return np.tanh(1 * x)


def der_tanh_act(x):
    return 1 / ((np.cosh(x)) ** 2)

class MultiLayerPerceptron:
    def __init__(self, inputNeuronas, hiddenNeuronas, outputNeuronas, epochs, epsilon, activation_function = sigmoid, der_activation_function = der_sigmoid):
        self.inputNeuronas = inputNeuronas
        self.output_size = outputNeuronas
        self.layers = []
        self.epochs = epochs
        self.epsilon = epsilon
        previous_layer_size = inputNeuronas
        #crea cada layer
        for layer_size in hiddenNeuronas:
            self.layers.append(NeuronLayer(previous_layer_size, layer_size, activation_function, der_activation_function))
            previous_layer_size = layer_size
        self.layers.append(NeuronLayer(previous_layer_size, outputNeuronas, activation_function, der_activation_function))

    def train(self, inputs, targets, eta):
        c = 0
        while c in range(self.epochs):
            total_error = 0
            for x, y in (inputs, targets):
                predicted = self.predict(inputs)
                error = np.mean((predicted - y) ** 2) / 2
                total_error += error
                delta = predicted - y
                for layer in reversed(self.layers):
                    delta = layer.backward(delta, eta)
            if total_error < self.epsilon:
                print(f"{c}")
                break
            elif c % 100 == 0:
                print(f"error: {total_error}")
            c +=1
            
    def predict(self, inputs_data):
        #revisar!!!! me parece que seria distinto, sino no llegaria nunca
        activations = inputs_data
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations
                
class NeuronLayer:
    def __init__(self, inputNeuronas, outputNeuronas, activation_f, der_activation_f):
        self.inputNeuronas = inputNeuronas
        self.outputNeuronas = outputNeuronas
        self.activation_function = activation_f
        self.der_activation_function = der_activation_f
        # inicializa con pequeños pesoss los w
        self.weights = np.random.rand(inputNeuronas + 1, outputNeuronas) * 0.1 - 0.05  
        
    def backward():
        # hace el back
        return
    
    def forward():
        # hace el forward
        return