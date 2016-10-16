package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.OutputLayerModel;
import com.github.mkorman9.neural.data.Vector;

class OutputLayerWeightsComputer {
    private OutputLayerModel outputLayerModel;
    private double learningRate;

    public OutputLayerWeightsComputer(OutputLayerModel outputLayerModel, double learningRate) {
        this.outputLayerModel = outputLayerModel;
        this.learningRate = learningRate;
    }

    public Vector compute(Vector hiddenLayerOutputs, double dv) {
        Vector newOutputLayerWeights = Vector.zero(hiddenLayerOutputs.size());
        for (int i = 0; i < hiddenLayerOutputs.size(); i++) {
            double value = computeNewValue(outputLayerModel.getWeights().get(i), learningRate, dv, hiddenLayerOutputs.get(i));
            newOutputLayerWeights.set(i, value);
        }
        return newOutputLayerWeights;
    }

    private double computeNewValue(double weight, double learningRate, double dv, double answer) {
        return weight + (learningRate * dv * answer);
    }
}
