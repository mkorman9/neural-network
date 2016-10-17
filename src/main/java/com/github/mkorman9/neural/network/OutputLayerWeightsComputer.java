package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Model;

class OutputLayerWeightsComputer {
    private Model networkModel;
    private double learningRate;

    public OutputLayerWeightsComputer(Model networkModel, double learningRate) {
        this.networkModel = networkModel;
        this.learningRate = learningRate;
    }

    public Matrix compute(Matrix dv) {
        Matrix newWeights = Matrix.zero(networkModel.getHiddenLayerNeuronsCount(), networkModel.getOutputLayerNeuronsCount());
        for (int i = 0; i < networkModel.getOutputLayerNeuronsCount(); i++) {
            for (int j = 0; j < networkModel.getHiddenLayerNeuronsCount(); j++) {
                newWeights.setValue(j, i, networkModel.getOutputLayerModel().getWeights().value(j, i) - learningRate * dv.value(j, i));
            }
        }
        return newWeights;
    }
}
