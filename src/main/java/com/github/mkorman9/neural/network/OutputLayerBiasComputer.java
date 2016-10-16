package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.OutputLayerModel;

class OutputLayerBiasComputer {
    private OutputLayerModel outputLayerModel;
    private double learningRate;

    public OutputLayerBiasComputer(OutputLayerModel outputLayerModel, double learningRate) {
        this.outputLayerModel = outputLayerModel;
        this.learningRate = learningRate;
    }

    public double compute(double dv) {
        return outputLayerModel.getBias() + (learningRate * dv);
    }
}
