
import numpy as np

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from util.activation_functions import Activation

from sklearn.metrics import accuracy_score

import random

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='ce', learningRate=0.005, epochs=50):

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
        elif loss == 'ce':
            self.loss = CrossEntropyError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance 9.98977852e-01of each epoch for later usages
        # e.g. plotting, reporting..
        self.train_perform= []
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer, optimal dropout for an input layer: 20% according to Nitish Srivastava et.al.
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, 
                           None, inputActivation, False, dropout=25, w_limit=4))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10,
                           None, outputActivation, True, w_limit=4))

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
        self.layers[0].forward(inp)
        self.layers[1].inp = np.insert(self.layers[0].outp, 0, 1)
        self.memory['layer2'].append(self.layers[1].forward(self.layers[1].inp))

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        return self.loss.calculateError(target, self.memory['layer2'])

    def _update_weights(self, learningRate, delta_input, delta_output):
        self.layers[1].updateWeights(learningRate, delta_output)
        self.layers[0].updateWeights(learningRate, delta_input)


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

            if "layer2" in self.memory:
                del self.memory['layer2'][:]
            else:
                self.memory['layer2'] = []

            r = list(zip(self.trainingSet.input, self.trainingSet.label))
            random.shuffle(r)
            self.trainingSet.input, self.trainingSet.label = zip(*r)
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
                accuracy = accuracy_score(np.array(self.trainingSet.label),
                                          self.transform(self.evaluate(self.trainingSet.input)))
                self.train_perform.append(accuracy)

    def _train_one_epoch(self):

        for index, img in enumerate(self.trainingSet.input):
            #forwarding,output result stored in self.memory['layer2']
            self._feed_forward(img)
            # delta E_x / delta o_j
            delta_E = -self.loss.calculateDerivative(self.trainingSet.label[index], self.memory['layer2'][-1])
            #output layer: delta E / delta net
            delta_output = self.layers[1].computeDerivative(delta_E, np.ones(10))
            #input layer: delta E / delta net
            delta_input = self.layers[0].computeDerivative(delta_output, self.layers[1].weights)
            #set output of all droped out neurons to 0, which are by each weight update randomly decided
            dropped = [1] * (self.layers[0].nOut - self.layers[0].dropout)
            dropped += [0] * self.layers[0].dropout
            random.shuffle(dropped)
            self.layers[0].outp = self.layers[0].outp*np.array(dropped)
            # update weights
            self._update_weights(self.learningRate, delta_input, delta_output)

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        self._feed_forward(test_instance)

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
                outp = np.array(self.memory['layer2'][-1], ndmin=2)
                continue
            outp = np.concatenate((outp, np.array(self.memory['layer2'][-1], ndmin=2)))
        return outp

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)

