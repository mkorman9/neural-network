package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.activation.Function;
import com.github.mkorman9.neural.data.OutputLayerModel;
import com.github.mkorman9.neural.data.Vector;

class OutputLayerNeuronActivationComputer {
    private int dimension;
    private Function activationFunction;

    public OutputLayerNeuronActivationComputer(int dimension, Function activationFunction) {
        this.dimension = dimension;
        this.activationFunction = activationFunction;
    }

    public double compute(Vector neuronsAnswers, OutputLayerModel outputLayerModel) {
        double sum = computeWeightSum(neuronsAnswers, outputLayerModel.getWeights());
        return activationFunction.compute(sum + outputLayerModel.getBias());
    }

    private double computeWeightSum(Vector neuronsAnswers, Vector outputLayerWeights) {
        double sum = 0.0;
        for (int i = 0; i < dimension; i++) {
            sum += neuronsAnswers.get(i) * outputLayerWeights.get(i);
        }
        return sum;
    }
}
