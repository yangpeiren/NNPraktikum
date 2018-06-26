
import numpy as np

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from util.activation_functions import Activation

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.005, epochs=50):

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
        self.outputActivation = outputActivation
        self.cost = []
        self.memory = {}

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
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, 
                           None, inputActivation, False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10, 
                           None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets

        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        self.memory['layer1'] = self.layers[0].forward(inp)
        temp = np.insert(self.memory['layer1'], 0, 1)
        self.memory['layer2'] = self.layers[1].forward(temp)

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        return self.loss.calculateError(target, self.memory['layer2'])
    
    def _update_weights(self, learningRate, current_index):
        """
        Update the weights of the layers by propagating back the error
        """
        error = self._compute_error(self.trainingSet.label[current_index])
        self.memory['derivatives2'] = self.layers[1].computeDerivative(error, np.ones(10))
        self.layers[1].updateWeights(learningRate)
        self.layers[0].computeDerivative(self.memory['derivatives2'], self.layers[1].weights.T)
        self.layers[0].updateWeights(learningRate)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}..".format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:

                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                accuracy = accuracy_score(self.validationSet.label,
                                          self.transform(self.evaluate(self.validationSet.input)))
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def _train_one_epoch(self):

        for index, img in enumerate(self.trainingSet.input):

            # Do a forward pass to calculate the output and the error
            self._feed_forward(img)

            # Update weights in the online learning fashion
            self._update_weights(self.learningRate, index)

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        self._feed_forward(test_instance)

    def calcul_score(self, labels, outps):
        """ TODO: whatever
        ----
        label: a matrix
        outp: a matrix
        ----
        RETURN: a float
        """
        score = 0
        for label, outp in zip(labels, outps):
            score += np.sum(np.square(label-outp))
        score = np.sqrt(np.divide(score, labels.shape[0]))
        return 1.0 - score

    def transform(self, outps):
        """

        :param outps: a matrix
        :return: a matrix that transform the input into a binary matrix that marks
                 the largest element as 1 for each row
        """
        bin_outps = np.zeros((outps.shape[0], outps.shape[1]))
        for index, outp in enumerate(outps):
            label_index = np.argmax(outp)
            bin_outps[index][label_index] = 1
        print(bin_outps)
        return bin_outps

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        outp = None
        for img in test:
            self.classify(img)
            if outp is None:
                outp = np.array(self.memory['layer2'], ndmin=2)
                continue
            outp = np.concatenate((outp, np.array(self.memory['layer2'], ndmin=2)))
        return outp

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)

