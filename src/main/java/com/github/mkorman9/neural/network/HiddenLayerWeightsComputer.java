package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.*;

class HiddenLayerWeightsComputer {
    private Model networkModel;
    private double learningRate;

    public HiddenLayerWeightsComputer(Model networkModel, double learningRate) {
        this.networkModel = networkModel;
        this.learningRate = learningRate;
    }

    public Matrix compute(Vector inputRow, Vector hiddenLayerOutputs, Vector outputLayerOutput, Vector expectedOutput) {
        Matrix newHiddenLayerWeights = Matrix.zero(networkModel.getInputsCount(), networkModel.getHiddenLayerNeuronsCount());
        return networkModel.getHiddenLayerModel().getWeights();
    }
}
