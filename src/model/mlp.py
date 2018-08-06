
import numpy as np

from util.loss_functions import CrossEntropyError, BinaryCrossEntropyError, SumSquaredError, MeanSquaredError, \
    DifferentError, AbsoluteError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='crossentropy', learningRate=0.05, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        elif loss == 'crossentropy':
            self.loss = CrossEntropyError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Hidden Layer
        self.layers.append(LogisticLayer(train.input.shape[1], 64, None, "sigmoid", False))

        self.layers.append(LogisticLayer(64, 64, None, "sigmoid", False))

        # Output layer
        self.layers.append(LogisticLayer(64, 10, None, "softmax", True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        #self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1, axis=1)
        #self.validationSet.input = np.insert(self.validationSet.input, 0, 1, axis=1)
        #self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        activationValues = (inp-min(inp))/(max(inp)-min(inp))
        for layer in self.layers:
            activationValues = layer.forward(activationValues)
        
    def _compute_error(self, target):
        return  self.loss.calculateError(target,  self._get_output_layer().outp)

    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(learningRate)

    def _backpropagate(self, target):
        next_delta = self.loss.calculateDerivative(target, self._get_output_layer().outp)
        next_weights = np.ones(self._get_output_layer().nOut)

        for layer in reversed(self.layers):
            next_delta = layer.computeDerivative(next_delta, np.transpose(next_weights))
            next_weights = layer.weights

    def train(self, verbose=True):
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}..".format(epoch + 1, self.epochs))

            self.trainEpoch()

            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")


    def trainEpoch(self):
        for inp, target in zip(self.trainingSet.input,
                              self.trainingSet.label):
            targetVector = np.zeros(10)
            targetVector[target] = 1.0
            self._feed_forward(inp)
            self._backpropagate(targetVector)
            self._update_weights(self.learningRate)





    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        self._feed_forward(test_instance)
        return np.argmax(self._get_output_layer().outp)
        

    def evaluate(self, test=None):
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0, axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
