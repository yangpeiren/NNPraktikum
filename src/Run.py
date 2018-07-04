#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)

    # myStupidClassifier = StupidRecognizer(data.trainingSet,
    #                                       data.validationSet,
    #                                       data.testSet)
    #
    # myPerceptronClassifier = Perceptron(data.trainingSet,
    #                                     data.validationSet,
    #                                     data.testSet,
    #                                     learningRate=0.005,
    #                                     epochs=30)
    #
    # myLRClassifier = LogisticRegression(data.trainingSet,
    #                                     data.validationSet,
    #                                     data.testSet,
    #                                     learningRate=0.005,
    #                                     epochs=30)
    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.01,
                                        epochs=30)
    myMLPClassifier.train()
    
    # # Report the result #
    # print("=========================")
    # evaluator = Evaluator()
    #
    # # Train the classifiers
    # print("=========================")
    # print("Training..")
    #
    # print("\nStupid Classifier has been training..")
    # myStupidClassifier.train()
    # print("Done..")
    #
    # print("\nPerceptron has been training..")
    # myPerceptronClassifier.train()
    # print("Done..")
    #
    # print("\nLogistic Regression has been training..")
    # myLRClassifier.train()
    # print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    # stupidPred = myStupidClassifier.evaluate()
    # perceptronPred = myPerceptronClassifier.evaluate()
    # lrPred = myLRClassifier.evaluate()
    mlpPred = myMLPClassifier.transform(myMLPClassifier.evaluate())
    
    # Report the result
    print("=========================")
    #
    # print("Result of the stupid recognizer:")
    # #evaluator.printComparison(data.testSet, stupidPred)
    # evaluator.printAccuracy(data.testSet, stupidPred)
    #
    # print("\nResult of the Perceptron recognizer:")
    # #evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.testSet, perceptronPred)
    #
    # print("\nResult of the Logistic Regression recognizer:")
    # #evaluator.printComparison(data.testSet, lrPred)
    # evaluator.printAccuracy(data.testSet, lrPred)
    #
    print("\nResult of the Multilayer Perceptron recognizer:")
    accuracy = accuracy_score(data.testSet.label, mlpPred)
    print("Accuracy on validation: {0:.2f}%"
          .format(accuracy * 100))

    # # Draw
    plot = PerformancePlot("MLP -- Validation")
    plot.draw_performance_epoch(myMLPClassifier.performances,
                                myMLPClassifier.epochs)
    plot = PerformancePlot("MLP -- Training")
    plot.draw_performance_epoch(myMLPClassifier.train_perform,
                                myMLPClassifier.epochs)
    
if __name__ == '__main__':
    main()

