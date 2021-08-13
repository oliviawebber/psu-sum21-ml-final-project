import numpy as np
import cv2
from scipy.special import expit
import matplotlib.pyplot as plt
import csv

class AutoEncoder:
    def __init__(self, data, numHiddenNodes, beta, p, eta, momentum):
        # Shuffle data and split it into train and test sets in approximately a 75-25 split
        np.random.shuffle(data)
        cutoff = 3000
        self.data = data[:cutoff,:]
        self.test = data[cutoff:,:]

        # Auto-encoder parameters
        self.numHiddenNodes = numHiddenNodes
        self.numDatapoints, self.numInputs = data.shape
        self.beta = beta
        self.p = p
        self.eta = eta
        self.momentum = momentum

        # Initial weights are random between -0.05 and 0.05
        self.layerOneWeights = np.random.rand(self.numInputs+1, self.numHiddenNodes)*0.1-0.05
        self.layerTwoWeights = np.random.rand(self.numHiddenNodes+1, self.numInputs)*0.1-0.05

        # We use a momentum term, so setup arrays for storing weight changes
        self.prevLayerOneWeightChange = np.zeros(self.layerOneWeights.shape)
        self.prevLayerTwoWeightChange = np.zeros(self.layerTwoWeights.shape)

    def train(self):
        self.iteration = 0
        with open('results-n{}.csv'.format(self.numHiddenNodes), 'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train', 'test'])
            while self.iteration < 10:
                # Record error for each epoch to both file and screen
                print('Epoch {}'.format(self.iteration))
                trainError = self.error(self.data)
                testError = self.error(self.test)
                print('Train error: {}'.format(trainError))
                print('Test error: {}'.format(testError))
                writer.writerow([self.iteration, trainError, testError])
                print('Training...')      

                self.backward()

                
                # Save an image of the real and reconstructed fish to help visualize how accurate
                # the process is
                layerOne, layerTwo = self.forward(self.data[0])
                img = self.data[0].reshape((32,32))
                plt.imsave("train-{}-real-n{}.png".format(self.iteration, self.numHiddenNodes),img)
                img = layerTwo.reshape((32,32))
                plt.imsave("train-{}-recon-n{}.png".format(self.iteration, self.numHiddenNodes),img)

                layerOne, layerTwo = self.forward(self.test[0])
                img = self.test[0].reshape((32,32))
                plt.imsave("test-{}-real-n{}.png".format(self.iteration, self.numHiddenNodes),img)
                img = layerTwo.reshape((32,32))
                plt.imsave("test-{}-recon-n{}.png".format(self.iteration, self.numHiddenNodes),img)
                self.iteration += 1
        
            

    def error(self, data):
        # Comptues the error between the input data and reconstructed data
        _, layerTwo = self.fullForward(data)
        diff = layerTwo - data
        normError = 0
        for row in diff:
            normError += np.linalg.norm(row)
        
        return normError / data.shape[0]

    def backward(self):
        # Train the network, data is shuffled prior to training to avoid learning the order
        np.random.shuffle(self.data)
        numDataPoints = self.data.shape[0]

        # Saves the current length through the array, helpful for knowing how long training has left
        iterRange = range(0, numDataPoints, int(numDataPoints/10))
        currentDataPoint = 0
        for dataPoint in self.data:
            # Print a tick mark every 10% through the data
            if currentDataPoint in iterRange:
                print('#',end='')
            currentDataPoint += 1

            # Compute the full activation of the network, then convert it into an average activation
            # for every node in the hidden layer. This will be used in the sparsity term later
            layerOneFull, _ = self.fullForward(self.data)
            layerOneFull = layerOneFull[:,:-1]
            averageActivations = np.mean(layerOneFull, axis=0)

            # Run the network forward and compute sigma at the output layer
            layerOne, layerTwo = self.forward(dataPoint)
            layerTwoSigma = (dataPoint-layerTwo)*layerTwo*(1-layerTwo)

            # Compute the weight and sparsity terms for sigma at the hidden layer. The sparsity term is a KL divergence
            # sparsity term that penalizes nodes which have high average activation, i.e. they fire frequently
            weightTerm = np.dot(np.transpose(self.layerOneWeights[:-1,:]), layerTwoSigma)
            sparsityTerm = (self.p/(averageActivations+0.0000000000001)) - (1-self.p)/((1-averageActivations)+0.0000000000001)
            layerOneSigma = layerOne*(1-layerOne)*(weightTerm + self.beta*sparsityTerm)

            # Compute the weight updates at layer one and two
            x = np.insert(dataPoint, 0, 1)
            x = x.reshape((len(x),1))
            layerOneSigma = layerOneSigma.reshape((len(layerOneSigma),1))
            layerOneWeightChange = self.eta*np.dot(x, np.transpose(layerOneSigma)) + self.momentum*self.prevLayerOneWeightChange

            x = np.insert(layerOne, 0, expit(1))
            x = x.reshape((len(x),1))
            layerTwoSigma = layerTwoSigma.reshape((len(layerTwoSigma),1))
            layerTwoWeightChange = self.eta*np.dot(x, np.transpose(layerTwoSigma)) + self.momentum*self.prevLayerTwoWeightChange

            # Update weights and save them for momentum term
            self.layerOneWeights += layerOneWeightChange
            self.layerTwoWeights += layerTwoWeightChange

            self.prevLayerOneWeightChange = layerOneWeightChange
            self.prevLayerTwoWeightChange = layerTwoWeightChange
        print('')
        return

    def fullForward(self, x):
        # Computes a forward pass through the network for all data points, used for finding the average
        # activation of all nodes for sparsity term. A sigmoid is used as the squashing function
        ones = np.ones((x.shape[0],1))
        x = np.concatenate((x,ones),axis=1)
        layerOneActivations = expit(np.dot(x, self.layerOneWeights))

        layerOneActivations = np.concatenate((layerOneActivations,ones),axis=1)
        layerTwoActivations = expit(np.dot(layerOneActivations, self.layerTwoWeights))

        return (layerOneActivations, layerTwoActivations)

    def forward(self, x):
        # Computes a forward pass through the network for a single datapoint, use for training
        # A sigmoid is used as the squashing function
        x = np.insert(x, 0, 1)
        layerOneActivations = expit(np.dot(x, self.layerOneWeights))
        layerOneActivationsReturn = layerOneActivations.copy()

        layerOneActivations = np.insert(layerOneActivations, 0, 1)
        layerTwoActivations = expit(np.dot(layerOneActivations, self.layerTwoWeights))

        return (layerOneActivationsReturn, layerTwoActivations)

np.random.seed(1)

# Load in the data sets produced by the data cleaning script
data = np.load(open('data.npy','rb'))

numHiddenNodes = 1024
beta = 100.0
p = 0.01
eta = 0.01
momentum = 0.75
ae = AutoEncoder(data, numHiddenNodes, beta, p, eta, momentum)
ae.train()
