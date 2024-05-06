import numpy as np
from activation_functions import *

class MultiLayerPerceptron:
    def __init__(self, inputData, expectedOutput, epochs, epsilon, learningRate, hiddenLeyers, nodesInHiddenLayers):
        self.layers = []
        self.inputData = inputData
        self.expectedOutput = expectedOutput
        self.epochs = epochs
        self.epsilon = epsilon
        self.min = np.min(expectedOutput)
        self.max = np.max(expectedOutput)
        self.learningRate = learningRate
        self.hiddenLeyers = hiddenLeyers
        self.nodesInHiddenLayers = nodesInHiddenLayers

        #Crea la hidden layer (Layer(cantidadNeuronas, inputSize))
        self.layers.append(NeuronLayer((self.nodesInHiddenLayers), len(self.inputData[0])))
        
        # Crea la layer de salida (como la # de salida es 1 numero -> 1.)
        self.layers.append(NeuronLayer(1, self.nodesInHiddenLayers))
        
        #crea cada layer
        """ for layer_size in hiddenNeuronas:
            self.layers.append(NeuronLayer(previous_layer_size, layer_size, activation_function, der_activation_function))
            previous_layer_size = layer_size
        self.layers.append(NeuronLayer(previous_layer_size, outputNeuronas, activation_function, der_activation_function)) """
            
    def train(self):
        trainSize = len(self.inputData)
        ERROR = 1
        epoch = 0
        mse_errors = []
        
        while ERROR > self.epsilon and epoch < self.epochs:
            
            wfinal = []
            # PARA CADA NODO
            for i in range(trainSize):
                # Forward activation
                activations = self.activate(self.inputData[i])
                wfinal.append(activations[-1])

                # Calcula el error para el output layer
                self.layers[-1].error(self.expectedOutput[i] - wfinal[i], wfinal[i])
                
                # Backward propagation
                for i in range(len(self.layers) - 2, -1, -1):
                    inherit_layer = self.layers[i + 1]
                    self.layers[i].error(np.dot(inherit_layer.weights,inherit_layer.error_d), activations[i + 1])

                for i in range(len(self.layers)):
                    self.layers[i].delta(activations[i], self.learningRate)

            mse_errors.append(self.mid_square_error(wfinal, self.expectedOutput))
            ERROR = mse_errors[-1]
            
            if epoch % 100 == 0:
                print(f'Actual epoch: {epoch}') 
                
            epoch += 1

        test = self.predict(self.inputData, self.expectedOutput)
        
        self.mse = mse_errors[epoch - 1]
                
        return self.mse
    
    def train_k(self, k_index):
        ERROR = 1
        mse_errors = []
        
        # Se mezclan aleatoriamente los datos de entrada y los datos esperados 
        # Los datos mezclados se dividen en "k" grupos o "folds". 
        # Cada grupo contiene una porción de los datos originales.
        # Orden deseado de los índices
        orden_indices = np.random.permutation(10)
        
        # Reorganizar los vectores de entrada de acuerdo con el nuevo orden
        entrada_ordenada = self.inputData[orden_indices]
        # Reorganizar el vector de salida de acuerdo con el nuevo orden
        salida_ordenada = self.expectedOutput[orden_indices]
    
        # Dividir el conjunto ordenado en 4 subconjuntos
        subconjuntos = np.array_split(entrada_ordenada, k_index)
        subconjuntos_salida = np.array_split(salida_ordenada, k_index)
        
        # Imprimir los subconjuntos
        print(subconjuntos)
        print(subconjuntos_salida)
        
        # entreno el modelo por todos los grupos excepto el actual que lo uso como prueba
        for j in range(k_index):
            epoch = 0
            input_train = np.concatenate([subconjuntos[i] for i in range(k_index) if i != j])
            expected_train = np.concatenate([subconjuntos_salida[i] for i in range(k_index) if i != j])
            trainSize = len(input_train)
            while ERROR > self.epsilon and epoch < self.epochs:
                
                wfinal = []
                # PARA CADA NODO
                for i in range(trainSize):
                    # Forward activation
                    activations = self.activate(input_train[i])
                    wfinal.append(activations[-1])

                    # Calcula el error para el output layer
                    self.layers[-1].error(expected_train[i] - wfinal[i], wfinal[i])
                    
                    # Backward propagation
                    for i in range(len(self.layers) - 2, -1, -1):
                        inherit_layer = self.layers[i + 1]
                        self.layers[i].error(np.dot(inherit_layer.weights,inherit_layer.error_d), activations[i + 1])

                    for i in range(len(self.layers)):
                        self.layers[i].delta(activations[i], self.learningRate)

                mse_errors.append(self.mid_square_error(wfinal, expected_train))
                ERROR = mse_errors[-1]
                
                if epoch % 100 == 0:
                    print(f'Actual epoch: {epoch}') 
                    
                epoch += 1
            
            test = self.predict(subconjuntos[i], subconjuntos_salida[i])

            self.mse = mse_errors[epoch - 1]
                
        return self.mse
    
    
    def activate(self, init_input):
        activations = [init_input]
        
        for layer in range(len(self.layers)):
            activations.append(self.layers[layer].activate(activations[-1]))
        return activations
            
    def predict(self, inputs_data, expected_data):
        #revisar!!!! me parece que seria distinto, sino no llegaria nunca
        #salida por cada perceptron
        predicted = 0
        outputValuesExpected = [-1, 1]
        for i in range(len(inputs_data)):
            activations = self.activate(inputs_data[i])
            
            print(f'output value: {activations[-1][0]}')
            # comparo cual de los dos valores obtuvo la red
            j =  (np.abs(outputValuesExpected-activations[-1][0])).argmin()
            
            #verifica si coincide lo expulsado con lo esperado
            if outputValuesExpected[j] == expected_data[i]:
                predicted += 1 
        
        print(f'acertados: {predicted}')
        
        #calculo el error
        w = []
        for i in range(len(inputs_data)):
            activations = self.activate(inputs_data[i])
            w.append(activations[-1])
        predict_mse = self.mid_square_error(w, expected_data)
        
        print(f'mse: {predict_mse}')
    
    def backward_propagation(self, predicted, expected, learningRate):
        delta = expected - predicted
        for layer in reversed(self.layers):
            delta = layer.backward_propagation(delta, predicted, learningRate)
        return delta
    
    def mid_square_error(self, w, expectedOutput):
        error = 0
        size = len(w)
        for i in range(size):
            error += (expectedOutput[i] - self.denormalize(w[i])) ** 2
        return np.sum(error) / size
    
    def denormalize(self, values):
        return ((values + 1) * (self.max - self.min) * 0.5) + self.min
    
    
class NeuronLayer:
    def __init__(self, neuronsQty, inputSize):
        self.neuronsQty = neuronsQty
        self.inputSize = inputSize
        self.bias = 1
        # inicializa con pequeños pesoss los w
        self.weights = np.random.default_rng().random((inputSize, neuronsQty))
        
    def activate(self, input):
        h = np.dot(input, self.weights) + self.bias
        return 1 / (1 + np.exp(-2 * 1 * h))
    
    def error(self, inheritedError, activation):
        d = (1 - activation ** 2) * self.bias
        self.error_d = d * inheritedError
        
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
        
    def delta(self, lastActivation, learningRate):
        actualMatrix = np.matrix(lastActivation)
        errorMatrix = np.matrix(self.error_d)
        self.update_weights(actualMatrix, errorMatrix, learningRate)

    def update_weights(self, actualMatrix, errorMatrix, learningRate):
        self.bias += learningRate * self.error_d
        self.weights += learningRate * np.dot(actualMatrix.T, errorMatrix)
        
       