import numpy as np
from scipy.special import expit

class AutoEncoder:
    def __init__(self, data, numHiddenNodes, beta, p, eta, momentum):
        # Initial setup
        np.random.shuffle(data)
        data = data[:4000,:]
        self.numHiddenNodes = numHiddenNodes
        self.data = data
        self.numDatapoints, self.numInputs = data.shape

        self.layerOneWeights = np.random.rand(self.numInputs+1, self.numHiddenNodes)*0.1-0.05
        self.layerTwoWeights = np.random.rand(self.numHiddenNodes+1, self.numInputs)*0.1-0.05

        self.beta = beta
        self.p = p
        self.eta = eta
        self.momentum = momentum

        self.prevLayerOneWeightChange = np.zeros(self.layerOneWeights.shape)
        self.prevLayerTwoWeightChange = np.zeros(self.layerTwoWeights.shape)

    def train(self):
        # Iterate backprop until 50 epochs have passed
        epoch = 0
        while epoch < 50:
            print('Epoch {}'.format(epoch))
            print(self.error())
            print('Training...')
            self.backward()
            epoch += 1
            

    def error(self):
        # Compute the error on the data set
        _, layerTwo = self.fullForward(self.data)
        diff = layerTwo - self.data
        normError = 0
        for row in diff:
            normError += np.linalg.norm(row)
        
        return normError / self.data.shape[0]

    def backward(self):
        # Perform backprop to update the network
        np.random.shuffle(self.data)
        numDataPoints = self.data.shape[0]
        iterRange = range(0, numDataPoints, int(numDataPoints/10))
        currentDataPoint = 0

        # Sequentially process each data point
        for dataPoint in self.data:
            if currentDataPoint in iterRange:
                print('#',end='')
            currentDataPoint += 1

            # Compute activations
            layerOneFull, _ = self.fullForward(self.data)
            layerOneFull = layerOneFull[:,:-1]
            averageActivations = np.mean(layerOneFull, axis=0)

            # Compute sigmas, only layerOne has a sparsity term associated with it
            layerOne, layerTwo = self.forward(dataPoint)
            layerTwoSigma = (dataPoint-layerTwo)*layerTwo*(1-layerTwo)
            
            weightTerm = np.dot(np.transpose(self.layerOneWeights[:-1,:]), layerTwoSigma)
            sparsityTerm = (self.p/(averageActivations+0.0000000000001)) - (1-self.p)/((1-averageActivations)+0.0000000000001)
            layerOneSigma = layerOne*(1-layerOne)*(weightTerm + self.beta*sparsityTerm)


            # Update the weights
            x = np.insert(dataPoint, 0, 1)
            x = x.reshape((len(x),1))
            layerOneSigma = layerOneSigma.reshape((len(layerOneSigma),1))
            layerOneWeightChange = self.eta*np.dot(x, np.transpose(layerOneSigma)) + self.momentum*self.prevLayerOneWeightChange

            x = np.insert(layerOne, 0, expit(1))
            x = x.reshape((len(x),1))
            layerTwoSigma = layerTwoSigma.reshape((len(layerTwoSigma),1))
            layerTwoWeightChange = self.eta*np.dot(x, np.transpose(layerTwoSigma)) + self.momentum*self.prevLayerTwoWeightChange

            self.layerOneWeights += layerOneWeightChange
            self.layerTwoWeights += layerTwoWeightChange

            # Save weights for momentum term
            self.prevLayerOneWeightChange = layerOneWeightChange
            self.prevLayerTwoWeightChange = layerTwoWeightChange
        print('')
        return

    def fullForward(self, x):
        # Run the network forward on every data point
        ones = np.ones((x.shape[0],1))
        x = np.concatenate((x,ones),axis=1)
        layerOneActivations = expit(np.dot(x, self.layerOneWeights))

        layerOneActivations = np.concatenate((layerOneActivations,ones),axis=1)
        layerTwoActivations = expit(np.dot(layerOneActivations, self.layerTwoWeights))

        return (layerOneActivations, layerTwoActivations)

    def forward(self, x):
        # Run the network forward on a single data point
        x = np.insert(x, 0, 1)
        layerOneActivations = expit(np.dot(x, self.layerOneWeights))
        layerOneActivationsReturn = layerOneActivations.copy()

        layerOneActivations = np.insert(layerOneActivations, 0, 1)
        layerTwoActivations = expit(np.dot(layerOneActivations, self.layerTwoWeights))

        return (layerOneActivationsReturn, layerTwoActivations)

np.random.seed(1)

# Load in the data sets produced by the data cleaning script
trainingLabels = np.load(open('mnist_train_label.npy','rb'))
trainingData = np.load(open('mnist_train_data.npy','rb'))

testLabels = np.load(open('mnist_test_label.npy','rb'))
testData = np.load(open('mnist_test_data.npy','rb'))

numHiddenNodes = 800
beta = 2.0
p = 0.05
eta = 0.01
momentum = 0.9
ae = AutoEncoder(trainingData, numHiddenNodes, beta, p, eta, momentum)
ae.train()
