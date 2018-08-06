import time

import numpy as np

from util.activation_functions import Activation


class LogisticLayer():

    def __init__(self, nIn, nOut, weights=None,
                 activation='sigmoid', isClassifierLayer=False):

        # Get activation function from string
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.activationDerivative = Activation.getDerivative(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        self.inp = np.ndarray((nIn,1))
        self.outp = np.ndarray((nOut,1))
        self.deltas = np.zeros((nOut,1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = (rns.uniform(-1,1,size=(nIn, nOut)))
        else:
            assert(weights.shape == (nIn, nOut))
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def forward(self, inp):
        self.inp = inp
        self.outp = self._fire(inp)
        return self.outp

    def computeDerivative(self, next_derivatives, next_weights):
        outputDerivative = self.activationDerivative(self.outp)
        if self.isClassifierLayer:
            self.deltas = np.multiply(outputDerivative,next_derivatives)
        else:
            self.deltas = outputDerivative * np.dot(next_derivatives, next_weights)
        return self.deltas

    def updateWeights(self, learningRate):
        """
        Update the weights of the layer
        """

        # weight updating as gradient descent principle
        for neuron in range(0, self.nOut):
            self.weights[:, neuron] -= (learningRate * np.dot(self.deltas[neuron],self.inp))

    def _fire(self, inp):
        return self.activation(np.dot(inp, self.weights))

    def isOutputlayer(self):
        return self.isClassifierLayer