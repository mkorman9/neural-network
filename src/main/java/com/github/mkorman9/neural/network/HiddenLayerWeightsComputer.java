package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.*;

class HiddenLayerWeightsComputer {
    private Model networkModel;
    private double learningRate;

    public HiddenLayerWeightsComputer(Model networkModel, double learningRate) {
        this.networkModel = networkModel;
        this.learningRate = learningRate;
    }

    public Matrix compute(Vector inputRow, Vector dw) {
        Matrix newWeights = Matrix.zero(networkModel.getInputsCount(), networkModel.getHiddenLayerNeuronsCount());
        for (int i = 0; i < networkModel.getHiddenLayerNeuronsCount(); i++) {
            for (int j = 0; j < networkModel.getInputsCount(); j++) {
                double delta = DeltaNormalizer.normalize(inputRow.get(j) * dw.get(i));
                newWeights.setValue(j, i, networkModel.getHiddenLayerModel().getWeights().value(j, i) + (delta * learningRate));
            }
        }
        return newWeights;
    }
}
