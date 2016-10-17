package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.activation.Function;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;

class OutputLayerNeuronActivationComputer {
    private Model networkModel;
    private Function activationFunction;

    public OutputLayerNeuronActivationComputer(Model networkModel, Function activationFunction) {
        this.networkModel = networkModel;
        this.activationFunction = activationFunction;
    }

    public Vector compute(Vector hiddenLayerOutputs) {
        Vector neuronOutputs = Vector.zero(networkModel.getOutputLayerNeuronsCount());
        for (int i = 0; i < networkModel.getOutputLayerNeuronsCount(); i++) {
            double sum = computeWeightSum(hiddenLayerOutputs, i);
            double activation = activationFunction.compute(sum + networkModel.getOutputLayerModel().getBias().get(i));
            neuronOutputs.set(i, activation);
        }
        return neuronOutputs;
    }

    private double computeWeightSum(Vector hiddenLayerOutputs, int i) {
        double sum = 0;
        for (int j = 0; j < networkModel.getHiddenLayerNeuronsCount(); j++) {
            sum += hiddenLayerOutputs.get(j) * networkModel.getOutputLayerModel().getWeights().value(j, i);
        }
        return sum;
    }
}
