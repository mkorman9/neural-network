package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.DeltaNormalizer;
import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;

class HiddenLayerBiasComputer {
    private Model networkModel;
    private double learningRate;

    public HiddenLayerBiasComputer(Model networkModel, double learningRate) {
        this.networkModel = networkModel;
        this.learningRate = learningRate;
    }

    public Vector compute(Vector dw) {
        Vector newBias = Vector.zero(networkModel.getHiddenLayerNeuronsCount());
        for (int i = 0; i < networkModel.getHiddenLayerNeuronsCount(); i++) {
            double delta = DeltaNormalizer.normalize(dw.get(i));
            newBias.set(i, networkModel.getHiddenLayerModel().getBias().get(i) + (delta * learningRate));
        }
        return newBias;
    }
}
