package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.DeltaNormalizer;
import com.github.mkorman9.neural.data.Matrix;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;

class OutputLayerWeightsComputer {
    private Model networkModel;
    private double learningRate;

    public OutputLayerWeightsComputer(Model networkModel, double learningRate) {
        this.networkModel = networkModel;
        this.learningRate = learningRate;
    }

    public Matrix compute(Vector hiddenLayerOutput, Vector dv) {
        Matrix newWeights = Matrix.zero(networkModel.getHiddenLayerNeuronsCount(), networkModel.getOutputLayerNeuronsCount());
        for (int i = 0; i < networkModel.getOutputLayerNeuronsCount(); i++) {
            for (int j = 0; j < networkModel.getHiddenLayerNeuronsCount(); j++) {
                double delta = DeltaNormalizer.normalize(hiddenLayerOutput.get(j) * dv.get(i));
                newWeights.setValue(j, i, networkModel.getOutputLayerModel().getWeights().value(j, i) + (learningRate * delta));
            }
        }
        return newWeights;
    }
}
