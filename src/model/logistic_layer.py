import time

import numpy as np
from sklearn.preprocessing import minmax_scale

from util.activation_functions import Activation


class LogisticLayer():
    """
    A layer of neural

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    deltas : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='sigmoid', isClassifierLayer=False, dropout=0):

        # Get activation function from string
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.activationDerivative = Activation.getDerivative(
                                    self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        self.inp = np.ndarray((nIn+1, 1))
        self.inp[0] = 1
        self.outp = np.ndarray((nOut, 1))
        self.deltas = np.zeros((nOut, 1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nIn + 1, nOut))-0.5
        else:
            assert(weights.shape == (nIn + 1, nOut))
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape
        self.dropout = dropout

    def forward(self, inp):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        inp : ndarrayself.memory['derivatives2'] =
            a numpy array (nIn + 1,1) containing the input of the layer

        Change outp
        -------
        outp: ndarray
            a numpy array (nOut,1) containing the output of the layerself.memory['derivatives2'] =
        """

        # Here you have to implement the forward pass
        self.inp = inp
        outp = self._fire(inp)
        self.outp = outp

        return outp

    def computeDerivative(self, next_derivatives, next_weights):
        """
        Compute the derivatives (backward)self.memory['derivatives2'] =

        Parameters
        ----------
        next_derivatives: ndarray
            a numpy array containing the derivatives from next layer
        next_weights : ndarray
            a numpy array containing the weights from next layer
layers
        Change deltas
        -------
        deltas: ndarray
            a numpy array containing the partial derivatives on this layer
        """

        # Here the implementation of partial derivative calculation

        # In case of the output layer, next_weights is array of 1
        # and next_derivatives - the derivative of the error will be the errors
        # Please see the call of this method in LogisticRegression.
        # self.deltas = (self.outp *
        #              (1 - self.outp) *
        #               np.dot(next_derivatives, next_weights))

        # Or more general: output*(1-output) is the derivatives of sigmoid
        # (sigmoid_prime)
        # self.deltas = (Activation.sigmoid_prime(self.outp) *
        #                np.dot(next_derivatives, next_weights))

        # Or even more general: doesn't care which activation function is used
        # dado: derivative of activation function w.r.t the output
        dado = self.activationDerivative(self.outp)
        #self.deltas = (dado * np.dot(next_derivatives, next_weights))
        if self.isClassifierLayer:
            self.deltas = np.multiply(dado, np.multiply(next_derivatives, next_weights))
        else:
            #delete the bias in next_weights
            self.deltas = np.multiply(dado, np.dot(next_weights[1:], next_derivatives))
        # Or you can explicitly calculate the derivatives for two cases
        # Page 40 Back-propagation slides
        # if self.isClassifierLayer:
        #     self.deltas = (next_derivatives - self.outp) * self.outp * \
        #                   (1 - self.outp)
        # else:
        #     self.deltas = self.outp * (1 - self.outp) * \
        #                   np.dot(next_derivatives, next_weights)
        #Or you can have two computeDerivative methods, feel free to call
        # the other is computeOutputLayerDerivative or such.
        return self.deltas

    def updateWeights(self, learningRate, deltas):
        """
        Update the weights of the layer
        """
        for neuron in range(0, self.nOut):
            self.weights[:, neuron] += (learningRate *
                                        deltas[neuron]*
                                        self.inp*(1-self.dropout/self.nOut))
        return

    def _fire(self, inp):

        # planned to do a min_max scaling to prod to avoid overflow, for any format of inp
        raw_prod = np.dot(inp, self.weights)
        # scaled_prod = minmax_scale(raw_prod, feature_range=(-10, 10))

        return self.activation(raw_prod)
