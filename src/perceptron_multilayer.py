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
    def __init__(self, x, y, eta, inputNeuronas, hiddenNeuronas, outputNeuronas, epochs, epsilon, activation_function = sigmoid, der_activation_function = der_sigmoid):
        self.inputNeuronas = inputNeuronas
        self.output_size = outputNeuronas
        self.layers = []
        self.x = x
        self.y = y
        self.epochs = epochs
        self.epsilon = epsilon
        self.eta = eta
        previous_layer_size = inputNeuronas
        #crea cada layer
        for layer_size in hiddenNeuronas:
            self.layers.append(NeuronLayer(previous_layer_size, layer_size, activation_function, der_activation_function))
            previous_layer_size = layer_size
        self.layers.append(NeuronLayer(previous_layer_size, outputNeuronas, activation_function, der_activation_function))
            
    def train(self, learningRate=0.01):
        errors = []
        for epoch in range(self.epochs):
            total_error = 0
            for x, y in zip(self.x, self.y):
                predicted = self.predict(x.reshape(1, -1))
                error = np.mean((predicted - y) ** 2) / 2
                total_error += error
                delta = predicted - y
                for layer in reversed(self.layers):
                    delta = layer.backward_propagation(delta, self.eta, learningRate)
            if total_error < self.epsilon:
                print(f"Converged at epoch {epoch}")
                errors.append(error)
                break
            elif epoch % 200 == 0: #cada 200 epocas imprime el erro
                print(f"Epoch {epoch}, Error: {total_error}")
                
        return errors
                
            
    def predict(self, inputs_data):
        #revisar!!!! me parece que seria distinto, sino no llegaria nunca
        output = inputs_data
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output
    
    def backward_propagation(self, predicted, expected, learningRate):
        delta = expected - predicted
        for layer in reversed(self.layers):
            delta = layer.backward_propagation(delta, predicted, learningRate)
        return delta
                
class NeuronLayer:
    def __init__(self, inputNeuronas, outputNeuronas, activation_f, der_activation_f):
        self.inputNeuronas = inputNeuronas
        self.outputNeuronas = outputNeuronas
        self.activation_function = activation_f
        self.der_activation_function = der_activation_f
        # inicializa con pequeños pesoss los w
        self.weights = np.random.rand(outputNeuronas, inputNeuronas + 1)
        
    def backward_propagation(self, delta, inputs, learningRate):
        inputsBiased = np.insert(inputs, 0, 1)
        
        #Calcula el gradiente
        delta = delta * der_sigmoid(np.dot(self.weights.T, inputsBiased))
        #actializa los pesos
        actualizedWeight = np.outer(delta, inputsBiased)
        self.weights += learningRate * actualizedWeight
        return np.dot(self.weights[:, 1:].T, delta)
    
    def forward_propagation(self, inputs):
        # agrega el sesgo.
        inputsBiased = np.insert(inputs, 0,1)
        
        #calcula la salida de la capa aplicando la funcion de activacion a la suma
        weightedSum = np.dot(self.weights, inputsBiased)
        activatedOutput = sigmoid(weightedSum)
        return activatedOutput