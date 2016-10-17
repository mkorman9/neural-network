package com.github.mkorman9.neural.network;

import com.github.mkorman9.neural.data.Model;
import com.github.mkorman9.neural.data.Vector;

class OutputLayerBiasComputer {
    private Model networkModel;
    private double learningRate;

    public OutputLayerBiasComputer(Model networkModel, double learningRate) {
        this.networkModel = networkModel;
        this.learningRate = learningRate;
    }

    public Vector compute(Vector dv) {
        Vector newBias = Vector.zero(networkModel.getOutputLayerNeuronsCount());
        for (int i = 0; i < networkModel.getOutputLayerNeuronsCount(); i++) {
            newBias.set(i, networkModel.getOutputLayerModel().getBias().get(i) + (learningRate * dv.get(i)));
        }
        return newBias;
    }
}
